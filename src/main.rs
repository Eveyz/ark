use std::io::Write;
use std::sync::Arc;

use arrow_array::{FixedSizeListArray, Float32Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use futures::StreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use thiserror::Error;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use llama_cpp_2::llama_batch::LlamaBatch;

#[derive(Error, Debug)]
pub enum ArkError {
    #[error("Database error: {0}")]
    Database(#[from] lancedb::error::Error),
    #[error("Embedding error: {0}")]
    Embedding(#[from] fastembed::Error),
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Llama error: {0}")]
    Llama(#[from] llama_cpp_2::LlamaCppError),
    #[error("Data not found: {0}")]
    NotFound(String),
    #[error("Unexpected error: {0}")]
    Unexpected(String),
}

type Result<T> = std::result::Result<T, ArkError>;

const DB_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/ark_db.lance");
const TABLE_NAME: &str = "survival_guide_md";
const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/model_cache/Qwen3-8B-Q4_K_M.gguf");
const CACHE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/model_cache/fastembed");

struct KnowledgeAgent {
    db: lancedb::Connection,
    embedder: TextEmbedding,
}

impl KnowledgeAgent {
    pub async fn new() -> Result<Self> {
        let db = lancedb::connect(DB_PATH).execute().await?;
        let cache_dir = std::path::Path::new(CACHE_DIR);
        let options = InitOptions::new(EmbeddingModel::BGESmallENV15)
            .with_show_download_progress(true)
            .with_cache_dir(cache_dir.to_path_buf());
        let embedder = TextEmbedding::try_new(options)?;
        let mut agent = Self { db, embedder };
        agent.init_database().await?;
        Ok(agent)
    }

    async fn init_database(&mut self) -> Result<()> {
        let table_names = self.db.table_names().execute().await?;
        if table_names.contains(&TABLE_NAME.to_string()) {
            return Ok(());
        }
        let mock_data = vec![
            "野外急救：如果队友小腿被蛇咬伤，绝对不可以切开伤口吸毒，这样会加速毒素吸收，且口中有伤口会使施救者中毒。正确的做法是：让伤者保持平静，避免剧烈活动；在伤口近心端用绷带包扎，减缓血液回流；尽量记住蛇的特征，并火速寻求专业医疗救助。",
        ];
        let embeddings = self.embedder.embed(mock_data.clone(), None)?;
        let dim = embeddings[0].len() as i32;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("source", DataType::Utf8, false),
            Field::new("h1", DataType::Utf8, false),
            Field::new("h2", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("vector", DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim), false),
        ]));

        let id_array = Int32Array::from(vec![1]);
        let source_array = StringArray::from(vec!["mock_source"]);
        let h1_array = StringArray::from(vec![""]);
        let h2_array = StringArray::from(vec![""]);
        let text_array = StringArray::from(mock_data.clone());
        let flat_embeddings: Vec<f32> = embeddings.into_iter().flatten().collect();
        let vector_array = Float32Array::from(flat_embeddings);
        let fixed_size_list_array = FixedSizeListArray::new(Arc::new(Field::new("item", DataType::Float32, true)), dim, Arc::new(vector_array), None);

        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(id_array), 
            Arc::new(source_array), 
            Arc::new(h1_array), 
            Arc::new(h2_array), 
            Arc::new(text_array), 
            Arc::new(fixed_size_list_array)
        ])?;
        let batches = arrow_array::RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema.clone());
        self.db.create_table(TABLE_NAME, Box::new(batches) as Box<dyn arrow_array::RecordBatchReader + Send>).execute().await?;
        Ok(())
    }

    pub async fn retrieve(&mut self, query: &str) -> Result<String> {
        // BGE English models recommend this prefix for retrieval
        let query_with_prefix = format!("Represent this sentence for searching relevant passages: {}", query);
        let mut query_embedding = self.embedder.embed(vec![query_with_prefix], None)?;
        let query_vec = query_embedding.pop().ok_or_else(|| ArkError::Unexpected("Failed to generate embedding".to_string()))?;
        let table = self.db.open_table(TABLE_NAME).execute().await?;
        
        let mut results = table.query().nearest_to(query_vec)?.limit(5).execute().await?;
        
        let mut contexts = Vec::new();
        if let Some(batch_res) = results.next().await {
            let batch: RecordBatch = batch_res?;
            if batch.num_rows() > 0 {
                let text_array = batch.column_by_name("text").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
                for i in 0..batch.num_rows() {
                    contexts.push(text_array.value(i).to_string());
                }
            }
        }
        
        if contexts.is_empty() {
            Err(ArkError::NotFound("No relevant data found in DB".to_string()))
        } else {
            Ok(contexts.join("\n...\n"))
        }
    }
}

/// 本地大模型推理引擎 (Llama-cpp-2 绑定，支持 Metal 加速)
struct LocalLlmAgent {
    backend: LlamaBackend,
    model: LlamaModel,
}

impl LocalLlmAgent {
    pub fn new(model_path: &str) -> Result<Self> {
        println!("[LocalLlmAgent] 正在初始化 LlamaBackend...");
        // 初始化底层硬件引擎 Backend，对于启用了 metal feature 的编译，这里会自动挂载 Metal 加速。
        let mut backend = LlamaBackend::init().map_err(|e| ArkError::Unexpected(e.to_string()))?;
        
        // 屏蔽所有底层的 ggml 和 Metal 编译日志输出，让终端保持干净
        backend.void_logs();
        
        println!("[LocalLlmAgent] 正在加载 GGUF 模型 (Metal 加速已开启)...");
        let model_params = LlamaModelParams::default();
        // 根据模型路径加载模型权重结构。
        let model = LlamaModel::load_from_file(&backend, std::path::Path::new(model_path), &model_params).map_err(|e| ArkError::Unexpected(e.to_string()))?;
        
        Ok(Self { backend, model })
    }

    /// 真正的 Agentic ReAct 回路：让模型自行决定何时搜索，搜什么，何时结束
    pub async fn react_chat(&self, query: &str, agent: &mut KnowledgeAgent) -> Result<()> {
        let system_prompt = r#"你是一个“方舟生存自动化专家”。你有能力主动查阅《野外生存手册》知识库来回答问题。
你可以使用以下工具：
- Check_Index: 查看所有可用的生存手册书目名称（当你不知道有哪些书或找不准关键词时使用）。Action Input 可以为空。
- Search: 在知识库中检索相关内容。搜索词强烈建议使用简短的英文关键词或短句。

你必须严格遵守以下执行流程（ReAct模式）：
Thought: 思考你需要做什么，分析当前已知信息。
Action: 决定使用的工具（只能填 Check_Index 或 Search，或者留空）。
Action Input: 传入工具的英文搜索关键词（如果是 Check_Index 则为空）。
Observation:

注意：当你输出完 "Observation:" 时，必须立刻停止！系统会拦截并在这里插入结果给你。你拿到结果后，可以继续新的 Thought。如果信息不够，你可以多次使用工具。
当你找齐了所有需要的信息，可以回答用户时，使用如下格式输出最终答案：
Thought: 我已经找齐了所需信息。
Final Answer: 详细的、基于查到事实的中文解答。

绝对不要自己编造外部知识！"#;

        let prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n问题：{}<|im_end|>\n<|im_start|>assistant\n",
            system_prompt, query
        );

        // 初始化 Llama Context，设置窗口大小为 8192 应对多轮 ReAct
        let mut ctx_params = LlamaContextParams::default();
        ctx_params = ctx_params.with_n_ctx(std::num::NonZeroU32::new(8192));
        let mut ctx = self.model.new_context(&self.backend, ctx_params).map_err(|e| ArkError::Unexpected(e.to_string()))?;

        println!("\n=== 方舟 Agent 启动自主推理回路 ===");

        let mut n_cur = 0;
        let tokens = self.model.str_to_token(&prompt, AddBos::Always).map_err(|e| ArkError::Unexpected(e.to_string()))?;
        let mut batch = LlamaBatch::new(8192, 1);
        let tokens_len = tokens.len();
        for (i, token) in tokens.into_iter().enumerate() {
            batch.add(token, n_cur, &[0], i == tokens_len - 1).unwrap();
            n_cur += 1;
        }

        let mut n_logits = batch.n_tokens() - 1;
        let mut max_steps = 3;

        loop {
            ctx.decode(&mut batch).map_err(|e| ArkError::Unexpected(e.to_string()))?;
            batch.clear();

            let mut generated_step = String::new();
            let mut hit_observation = false;
            let mut hit_final = false;

            loop {
                let mut candidates = LlamaTokenDataArray::from_iter(ctx.candidates_ith(n_logits), false);
                let new_token_id = candidates.sample_token_greedy();

                let token_bytes = self.model.token_to_piece_bytes(new_token_id, 32, false, None).unwrap_or_default();
                let token_str = String::from_utf8_lossy(&token_bytes);

                if new_token_id == self.model.token_eos() || token_str.contains("<|im_end|>") {
                    hit_final = true;
                    break;
                }

                generated_step.push_str(&token_str);
                print!("{}", token_str);
                std::io::stdout().flush()?;

                batch.clear();
                batch.add(new_token_id, n_cur, &[0], true).unwrap();
                n_cur += 1;

                ctx.decode(&mut batch).map_err(|e| ArkError::Unexpected(e.to_string()))?;
                n_logits = 0; // The next candidates_ith will look at the 0th index of the single-token batch we just decoded

                if generated_step.ends_with("Observation:") {
                    hit_observation = true;
                    break;
                }
                
                if n_cur >= ctx.n_ctx() as i32 {
                    hit_final = true;
                    break;
                }
            }

            if hit_final {
                break;
            }

            if hit_observation {
                let mut action_type = String::new();
                if let Some(action_idx) = generated_step.rfind("Action:") {
                    let sub = &generated_step[action_idx + "Action:".len()..];
                    if let Some(nl) = sub.find('\n') {
                        action_type = sub[..nl].trim().to_string();
                    } else {
                        action_type = sub.trim().to_string();
                    }
                }

                let mut search_query = String::new();
                if let Some(action_idx) = generated_step.rfind("Action Input:") {
                    let sub = &generated_step[action_idx + "Action Input:".len()..];
                    let sub = sub.strip_suffix("Observation:").unwrap_or(sub);
                    search_query = sub.trim().to_string();
                }

                let obs_text = if max_steps == 0 {
                    println!(" [系统介入: 搜索次数已达上限强制停止]");
                    " \n[系统: 搜索次数已达上限，请立即根据已知信息使用 'Final Answer: ' 给出最终回答。绝对不要再调用工具！]\nThought: ".to_string()
                } else if action_type == "Check_Index" {
                    max_steps -= 1;
                    println!(" [系统介入: 查阅手册索引] (剩余次数: {})", max_steps);
                    let index_data = "可用手册目录:\n- ATP_3-50.21_Survival\n- Canadian_Cold_Weather\n- Canadian_Military_Fieldcraft\n- Deadfalls_and_Snares\n- FM21-76_SurvivalManual\n- FM_21-60_Visual_Signals\n- FM_21-76-1_Survival_Evasion_Recovery\n- FM_31-70_Basic_Cold_Weather\n- FM_4-25.11_First_Aid\n- SODIS_Safe_Water_Manual\n- ST_31-91B_SF_Medical_Handbook\n- Survival_Austere_Medicine\n- USMC_Winter_Survival\n- Where_There_Is_No_Doctor";
                    format!(" \n{}\nThought: ", index_data)
                } else if action_type == "Search" && !search_query.is_empty() {
                    max_steps -= 1;
                    println!(" [系统介入: 检索数据库 => {}] (剩余次数: {})", search_query, max_steps);
                    let obs_result = match agent.retrieve(&search_query).await {
                        Ok(res) => res,
                        Err(_) => "没有找到相关文档。".to_string(),
                    };
                    format!(" \n{}\nThought: ", obs_result)
                } else {
                    max_steps -= 1;
                    " \n[系统: 工具或参数缺失]\nThought: ".to_string()
                };

                print!("{}", obs_text);
                std::io::stdout().flush()?;
                
                let obs_tokens = self.model.str_to_token(&obs_text, AddBos::Never).unwrap();
                batch.clear();
                let obs_tokens_len = obs_tokens.len();
                
                if n_cur + obs_tokens_len as i32 >= ctx.n_ctx() as i32 {
                    println!("\n[系统警告: 上下文超限，强制终止推理]");
                    break;
                }
                
                for (i, token) in obs_tokens.into_iter().enumerate() {
                    batch.add(token, n_cur, &[0], i == obs_tokens_len - 1).unwrap();
                    n_cur += 1;
                }
                n_logits = batch.n_tokens() - 1;
            }
        }

        println!("\n\n========================");
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut knowledge_agent = KnowledgeAgent::new().await?;
    let llm_agent = LocalLlmAgent::new(MODEL_PATH)?;

    println!("\n=== 方舟生存指南 RAG 系统已启动 ===");
    println!("输入你的问题，或者输入 'quit', 'exit' 退出。");

    loop {
        print!("\n> ");
        std::io::stdout().flush()?;

        let mut user_input = String::new();
        std::io::stdin().read_line(&mut user_input)?;
        let query = user_input.trim();

        if query.is_empty() {
            continue;
        }

        if query.eq_ignore_ascii_case("quit") || query.eq_ignore_ascii_case("exit") {
            break;
        }

        // 这里不再单次检索，而是全权交给 ReAct 回路
        llm_agent.react_chat(query, &mut knowledge_agent).await?;
    }

    println!("退出系统。");
    Ok(())
}

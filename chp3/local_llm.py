import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer  # 改用 modelscope

model_id = "qwen/Qwen1.5-0.5B-Chat"  # 注意 modelscope 上的路径
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
print("模型和分词器加载完成！")

# 准备对话输入
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好，请介绍你自己。"}
]

# 使用分词器的模板格式化输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 编码输入文本
model_inputs = tokenizer([text], return_tensors="pt").to(device)

print("编码后的输入文本:")
print(model_inputs)
print(model_inputs.input_ids.shape)


# 使用模型生成回答
# max_new_tokens 控制了模型最多能生成多少个新的Token
# 内部会将token ids embedding 得到 token embeddings，
# 经过模型的前向传播得到输出 token ids
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)

# 将生成的 Token ID 截取掉输入部分
# 这样我们只解码模型新生成的部分
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 解码生成的 Token ID
# Token ID -> Token
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n模型的回答:")
print(response)

"""
使用 Hugging Face Transformers 的 model.generate() 进行文本生成示例。
与 example.py（nano-vllm）对比，展示同一任务用 transformers 原生 API 的写法。
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    # 使用聊天模板格式化（与 example.py 一致）
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    # 批量 tokenize
    model_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=model.config.max_position_embeddings,
    ).to(model.device)

    # 使用 transformers.generate 生成
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.6,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    # 只解码新生成的部分（去掉 prompt）
    for i, (text, output_ids) in enumerate(zip(texts, generated_ids)):
        input_len = model_inputs.input_ids[i].shape[0]
        new_token_ids = output_ids[input_len:]
        completion = tokenizer.decode(new_token_ids, skip_special_tokens=True)

        print("\n")
        print(f"Prompt: {text!r}")
        print(f"Completion: {completion!r}")


if __name__ == "__main__":
    main()

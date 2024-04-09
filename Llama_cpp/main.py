from llama_cpp import Llama

zephyr_model_path = "./models/zephyr-7b-beta.Q4_0.gguf"
context_size = 512

zephyr_model = Llama(model_path=zephyr_model_path, n_ctx=context_size)


def generate_zephyr(prompt, max_token=100, temperature=0.3, top_p=0.1, echo=True, stop=["Q", "\n"]):
    model_output = zephyr_model(prompt, max_tokens=max_token, temperature=temperature, top_p=top_p, echo=echo, stop=stop)
    return model_output

if __name__ == "__main__":
    prompt = "What do you think about the inclusion policies in Tech companies?"
    result = generate_zephyr(prompt)
    print(result)
    final_result = result["choices"][0]["text"].strip()
    print(final_result)

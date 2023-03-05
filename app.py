import os
import gradio as gr
import openai

API_TOKEN = os.getenv("API_TOKEN")

openai_engines = ["gpt-3.5-turbo", "text-davinci-003", "code-davinci-002", "text-curie-001"]
initial_prompt = "Let's discuss about digital future for the humans."


def chatgpt3(
        prompt,
        history,
        openai_token,
        engine,
        temperature,
        max_tokens,
        top_p,
        frequency_penalty,
        presence_penalty,
):
    history = history or []
    history_prompt = list(sum(history, ()))
    history_prompt.append(f"{prompt}")
    inp = " ".join(history_prompt)

    # keep the prompt length limited to ~2000 tokens
    inp = " ".join(inp.split()[-2000:])

    # remove duplicate sentences
    sentences = inp.split(".")
    sentence_dict = {}
    for i, s in enumerate(sentences):
        if s not in sentence_dict:
            sentence_dict[s] = i

    unique_sentences = [sentences[i] for i in sorted(sentence_dict.values())]
    inp = " ".join(unique_sentences)

    # create the output with openai
    openai.api_key = openai_token

    response = openai.ChatCompletion.create(
        model=engine,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )

    out = response.choices[0]['message']['content']
    history.append((inp, out))

    return history, out, ""


with gr.Blocks(title="ChatGPT vs ChatGPT") as block:
    gr.Markdown("## ChatGPT vs ChatGPT")
    with gr.Row():
        with gr.Column():
            openai_token = gr.Textbox(label="OpenAI API Key", value=API_TOKEN, visible=False)
            engine = gr.Dropdown(
                label="GPT3 Engine",
                choices=openai_engines,
                value="gpt-3.5-turbo",
            )
            temperature = gr.Slider(label="Temperature", minimum=0, maximum=1, step=0.1, value=0.9)
            max_tokens = gr.Slider(label="Max Tokens", minimum=10, maximum=400, step=10, value=150)
            top_p = gr.Slider(label="Top P", minimum=0, maximum=1, step=0.1, value=1)
            frequency_penalty = gr.Slider(
                label="Frequency Penalty",
                minimum=0,
                maximum=1,
                step=0.1,
                value=0,
            )
            presence_penalty = gr.Slider(
                label="Presence Penalty",
                minimum=0,
                maximum=1,
                step=0.1,
                value=0.6,
            )

        with gr.Column():
            message = gr.Textbox(value=initial_prompt, label="Type initial topic here:")
            chatbot = gr.Chatbot()
            state = gr.State()

            message.submit(
                chatgpt3,
                inputs=[
                    message,
                    state,
                    openai_token,
                    engine,
                    temperature,
                    max_tokens,
                    top_p,
                    frequency_penalty,
                    presence_penalty,
                ],
                outputs=[chatbot, message, state],
            )

if __name__ == "__main__":
    block.launch(debug=True)

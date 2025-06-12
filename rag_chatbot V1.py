# -*- coding: utf-8 -*-


from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

embedding = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-8B")
faiss_path = "/Users/maleksandrowicz/Desktop/internship2025/faiss_vector_base"
vectorstore = FAISS.load_local(faiss_path, embedding, allow_dangerous_deserialization=True)

model_name = "Qwen/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

verifier_model_name = "Qwen/Qwen2-1.5B-Instruct"
verifier_tokenizer = AutoTokenizer.from_pretrained(verifier_model_name, trust_remote_code=True)
verifier_model = AutoModelForCausalLM.from_pretrained(
    verifier_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

conversation_history = []


def verify_answer(question, context, generated_answer):

    verification_prompt_template = """You are an AI fact-checker. Your sole task is to determine if the 'Generated Answer' is factually supported by the 'Source Context'.

Rules:
1. The answer MUST NOT invent information or hallucinate details not present in the 'Source Context'.
2. The only exception is if the answer correctly states that it cannot answer the question ("I do not know", "I cannot find this information"). This is a valid response.

User Question: "{question}"
Source Context: "{context}"
Generated Answer: "{generated_answer}"

Is the 'Generated Answer' factually supported by the 'Source Context' according to the rules? Respond with only one word: VALID or INVALID.
"""
    
    verification_prompt = verification_prompt_template.format(
        question=question,
        context=context,
        generated_answer=generated_answer
    )

    inputs = verifier_tokenizer(verification_prompt, return_tensors="pt")
    device = verifier_model.device
    inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

    outputs = verifier_model.generate(**inputs_on_device, max_new_tokens=10, pad_token_id=verifier_tokenizer.eos_token_id)
    verification_result = verifier_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if "VALID" in verification_result.upper():
        return True
    else:
        return False

def ask_question(user_input):
    relevant_docs = vectorstore.similarity_search(user_input, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    base_prompt = """You are a professional and helpful assistant for the Bank of Cyprus.
If the information wasn't given in the context, state that you cannot answer the question, that you do not know, and advise where the user could find additional help and say that youdo not have this information.
Additionally, make sure that you answer in the language the user talks to you.
Make sure that there are no Chinese characters in your response.
Double-check your answer so that it satisfies the prompt and the needs of the user."""

    prompt_with_history = base_prompt
    for turn in conversation_history:
        prompt_with_history += f"\nUser: {turn['user']}\nAI: {turn['bot']}"
    prompt_with_history += f"\nUser: {user_input}\nRelevant info: {context}\nAI:"

    max_attempts = 3
    for attempt in range(max_attempts):
        
        inputs = tokenizer(prompt_with_history, return_tensors="pt")
        device = model.device
        inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs_on_device,
            max_new_tokens=500,
            temperature=0.7 + (attempt * 0.1),  
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        bot_reply = response.split("AI:")[-1].strip()

        if verify_answer(question=user_input, context=context, generated_answer=bot_reply):
            conversation_history.append({"user": user_input, "bot": bot_reply})
            return bot_reply, context
        else:
            print("Answer was invalid, regenerating...")

    print("\nFailed to generate a valid response after multiple attempts.")
    fallback_answer = "I apologize, but I am having trouble generating response based on the available information. Please try rephrasing or contact Bank of Cyprus support directly for assistance."
    conversation_history.append({"user": user_input, "bot": fallback_answer})
    return fallback_answer, context


print("Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    answer, context = ask_question(user_input)
    print(f"AI: {answer }\n")

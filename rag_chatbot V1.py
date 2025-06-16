# -*- coding: utf-8 -*-

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

embedding = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-8B")
faiss_path = "/Users/maleksandrowicz/Desktop/internship2025/faiss_vector_base"
vectorstore = FAISS.load_local(faiss_path, embedding, allow_dangerous_deserialization=True)

model_name = "Qwen/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

conversation_history = []

def verify_answer(question, context, generated_answer):
    verification_prompt_template ="""You are a verification AI auditor checking if the generated answer follows these guidelines:
1. The answer is grounded in the context and does not hallucinate.
2. The answer is not empty.
Please evaluate the answer below using ONLY the provided context

User Question: "{question}"
Source Context: "{context}"
Generated Answer: "{generated_answer}"

Your entire response MUST follow this exact format, with no other text:
STATUS: [VALID/INVALID] | BROKEN_RULE: [Rule number, or "N/A" if valid] | REASON: [Brief explanation of the failure, or "None" if valid]
"""
    
    verification_prompt = verification_prompt_template.format(
        question=question, context=context, generated_answer=generated_answer
    )

    inputs = tokenizer(verification_prompt, return_tensors="pt")
    device = model.device
    inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

    input_length = inputs_on_device["input_ids"].shape[1]
    outputs = model.generate(**inputs_on_device, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    new_tokens = outputs[0, input_length:]
    verification_result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    match = re.search(r"STATUS:\s*(VALID|INVALID)\s*\|\s*BROKEN_RULE:\s*([\w/]+)\s*\|\s*REASON:\s*(.*)", verification_result, re.DOTALL)

    if match:
        status, broken_rule, reason = match.groups()
        is_valid = status.strip() == "VALID"
        return (is_valid, None if is_valid else broken_rule.strip(), None if is_valid else reason.strip())
    else:
        return (False, "Format Error", f"Verifier response was malformed: '{verification_result}'")



def ask_question(user_input):
    relevant_docs = vectorstore.similarity_search(user_input, k=8)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    system_prompt_generator ="""You are a professional and helpful assistant for the Bank of Cyprus. 
    Your primary goal is to provide a complete and accurate answer to the user's question using ONLY the provided context.
    Strive for conciseness, but do not sacrifice necessary information. Your answer's length should adapt to the user's query;
    a simple question requires a short answer, while a complex question may require a more detailed one.    If the information isn't available, state that you don't know and advise where to find help.
    Answer in the user's language and do not add any meta-comments about your performance."""
    
    critique_for_next_attempt = ""
    max_attempts = 3
    for attempt in range(max_attempts):
        
        user_prompt_generator = f"""Based on the following context, please answer the user's question.
    Context:
    ---
    {context}
    ---
    User Question: {user_input}
    {critique_for_next_attempt}
    """
        
        messages = [
            {"role": "system", "content": system_prompt_generator},
            {"role": "user", "content": user_prompt_generator}
        ]
    
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs, max_new_tokens=400, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
        
        input_length = inputs["input_ids"].shape[1] 

        new_tokens = outputs[0, input_length:] 

        bot_reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        bot_reply = re.sub(r'<\|im_end\|>', '', bot_reply).strip()


        is_valid, broken_rule, reason = verify_answer(question=user_input, context=context, generated_answer=bot_reply)

        if is_valid:
            conversation_history.append({"user": user_input, "bot": bot_reply})
            return bot_reply, context
        else:
            print(f"DEBUG: Verification FAILED. Rule: {broken_rule}, Reason: {reason}. Regenerating...")
            critique_for_next_attempt = f"\nSYSTEM_CORRECTION: Your previous answer was invalid. It broke Rule #{broken_rule} - '{reason}'. Please provide a new, corrected answer that follows all rules."

    fallback_answer = "I apologize, but I am having trouble generating a factually supported response. Please contact Bank of Cyprus support for assistance."
    return fallback_answer, context
        
print("Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    answer, context = ask_question(user_input)
    print(f"AI: {answer }\n")



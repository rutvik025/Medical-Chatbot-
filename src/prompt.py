prompt_template = """
Use the given information context to give appropriate answer for the user's question.
If you don't know the answer, just say that you know the answer, but don't make up an answer.

Context: {context}
Question: {question}

Only return the appropriate answer and nothing else.
Helpful answer:
"""
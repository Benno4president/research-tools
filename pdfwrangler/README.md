## scipdf
- https://pypi.org/project/scipdf-parser/
- pip install git+https://github.com/titipata/scipdf_parser
- bash serve_grobid.sh

## 
langchain history works like this: llm = OpenAI(temperature=0)from langchain.prompts.prompt import PromptTemplatefrom langchain.chains import ConversationChaintemplate = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.Relevant Information:{history}Conversation:Human: {input}AI:"""prompt = PromptTemplate( input_variables=["history", "input"], template=template)conversation_with_kg = ConversationChain( llm=llm, verbose=True, prompt=prompt, memory=ConversationKGMemory(llm=llm))conversation_with_kg.predict(input="Hi, what's up?")
the library gpt4all can be used in python like this: 
```python3 
import gpt4all gptj = gpt4all.GPT4All("ggml-gpt4all-j-v1.3-groovy") 
messages = [{"role": "user", "content": "Name 3 colors"}] 
answer = gptj.chat_completion(messages) 
``` 
Using gpt4all via langchain and scipdf lib, write a python script to enhance a research paper in pdf format. Enhancement should be text marked by red if it poses a question or problem, text marked in yellow if it is evidence and text marked in green if it is a solution or a conclusion. Mark only single sentences. Include context/history to gpt4all, and remember to track it. Include a pre-text for  gpt4all promps. Output should be a new pdf file, identical to the original but include any modifications.

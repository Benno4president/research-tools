#https://python.langchain.com/en/latest/modules/models/llms/integrations/gpt4all.html
#https://python.langchain.com/en/latest/modules/memory/types/kg.html
#import gpt4all
import scipdf

#gptj = gpt4all.GPT4All("ggml-model-gpt4all-falcon-q4_0")
context = []
pre_text = "Classify this sentence from a research paper: "

def enhance_pdf(input_pdf, output_pdf):
    with scipdf.parse_pdf_to_dict(input_pdf) as pdf:
        for page in pdf:
            for sentence in page.sentences:
                messages = [{"role": "user", "content": pre_text + sentence.text}]
                messages.extend(context)
                answer = gptj.chat_completion(messages)
                context.append({"role": "bot", "content": answer})
                if "question" in answer or "problem" in answer:
                    sentence.mark("red")
                elif "evidence" in answer:
                    sentence.mark("yellow")
                elif "solution" in answer or "conclusion" in answer:
                    sentence.mark("green")
        pdf.save(output_pdf)

if __name__ == '__main__':
    import sys
    enhance_pdf(sys.argv[1], "output.pdf")



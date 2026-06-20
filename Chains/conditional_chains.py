from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from typing import Literal
from pydantic import BaseModel, Field


# Load model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)


# Output schema
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the feedback"
    )


# Parsers
parser = PydanticOutputParser(pydantic_object=Feedback)
parser2 = StrOutputParser()


# Prompt for sentiment classification
prompt1 = PromptTemplate(
    template="""
Determine whether the following feedback is positive or negative.

Feedback:
{feedback}

Answer with exactly one word and nothing else.

positive
negative
""",
    input_variables=["feedback"]
)
# Classification chain
classify_chain = prompt1 | llm | parser


# Prompts for response generation
prompt2 = PromptTemplate(
    template="""
Write an appropriate response to this positive feedback:

{feedback}
""",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="""
Write an appropriate response to this negative feedback:

{feedback}
""",
    input_variables=["feedback"]
)


# Branch chain
branch_chain = RunnableBranch(
    (
        lambda x: x["sentiment"].sentiment == "positive",
        prompt2 | llm | parser2
    ),
    (
        lambda x: x["sentiment"].sentiment == "negative",
        prompt3 | llm | parser2
    ),
    RunnableLambda(lambda x: "Could not determine sentiment")
)


# Combine both chains
chain = (
    RunnableLambda(
        lambda x: {
            "feedback": x["feedback"],
            "sentiment": classify_chain.invoke({"feedback": x["feedback"]})
        }
    )
    | branch_chain
)


# Test
result = chain.invoke(
    {"feedback": "This is a beautiful phone"}
)

print(result)

# Print graph
chain.get_graph().print_ascii()
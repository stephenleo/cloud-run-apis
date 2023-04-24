from enum import Enum
from typing import Optional

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field


class ModelName(str, Enum):
    gpt3p5 = "gpt3p5"
    gpt4 = "gpt4"


class RoleType(str, Enum):
    ai = "ai"
    human = "human"


class MessageData(BaseModel):
    content: str = Field(example="Hi! How can I help you today?")
    additional_kwargs: Optional[dict] = Field(default={}, example={})


class Message(BaseModel):
    type: RoleType = Field(example="ai")
    data: MessageData

    class Config:
        use_enum_values = True


def open_ai_model(open_ai_model_name: ModelName):
    if open_ai_model_name == ModelName.gpt3p5:
        return "gpt-3.5-turbo"
    elif open_ai_model_name == ModelName.gpt4:
        return "gpt-4"
    else:
        raise ValueError(f"Unknown model name: {open_ai_model_name}")


# Set up a parser + inject instructions into the prompt template.

# class BookResponse(BaseModel):
#     response: str
#     list_of_books: list[str]

# parser = PydanticOutputParser(pydantic_object=BookResponse)
# template = """{format_instructions}

# Query: {query}
# """

# human_prompt = PromptTemplate(
#     template=template,
#     input_variables=["query"],
#     partial_variables={"format_instructions": parser.get_format_instructions()},
# )

# query = lc_utils.human_prompt.format_prompt(query=query).to_string()
# response = parser.parse(response)

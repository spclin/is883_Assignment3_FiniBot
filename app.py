# Import necessary libraries
import streamlit as st
import openai
import os
from io import StringIO
import pandas as pd



# Make sure to add your OpenAI API key in the advanced settings of streamlit's deployment
open_AI_key = os.environ.get('OPENAI_API_KEY')
openai.api_key = open_AI_key



# UX goes here. You will have to incorporate some variables from the code above and make some tweaks.

st.set_page_config(page_title="Finibot", page_icon="🤖")
st.title("Finibot")

st.header("Hello! Welcome to the Financial Advisor Chatbot! To get started, please upload your CSV file (use [this template](https://drive.google.com/file/d/1-OjQbxRZqlTmTeU-KD6pW_9JQWrpCsTO/view?usp=sharing)).")

csv_file = st.file_uploader("upload file", type={"csv"})

if csv_file is not None:
    text_df = pd.read_csv(csv_file)
    st.write(text_df)
    
    # Convert the DataFrame to CSV and then to a string
    text_io = StringIO()
    text_df.to_csv(text_io, index=False)
    text = text_io.getvalue()
    text_io.close()

    return text

level = st.radio(
    "What's your level of expertise?",
    ["Novice", "Expert"],
    index=None,
)

st.write("You selected:", level)



### Here, with some adjustments, copy-paste the code you developed for Question 1 in Assignment 3 
##########################################################################

Output_template="""
Using the input, you will provide three separate sections of information.
First, show the "Total savings" (savings), "Monthly debt" (credit card debt), and "Monthly income" (income) in separate lines.
Second, with a blank line in between, add a new line called "Financial situation:", where below that line you will insert the financials of the user.
Finally, with a blank line in between, add a new line called "Recommendation:", where below that line you will insert the resulting route from routes (either an investment advisor or a debt advisor).
input: {input}
"""

investment_template ="""
Here you will commend the client's financial accomplishment and then turn into an investment advisor.
You will advise to invest their money and provide them with an investment portfolio based on their savings and using 5 stocks.
You will find the savings in {input}.
""" + Output_template

debt_template= """
Here you will politely and cautiously, without taking them on a guilt trip, warn the client about their financial situation.
You will then turn into a debt advisor, and create a plan for them to pay off their debt by allocating 10% of their income for monthly debt payments.
You will find the savings in {input}.
""" + Output_template

routes = [
	{
		"name": "investment advisor",
		"description": "Will be used if the debt ratio is less than 0.3",
		"prompt_template": investment_template,
	},
	{
		"name": "debt advisor",
		"description": "Will be used if the debt ratio is more than or equal to 0.3",
		"prompt_template": debt_template,
	},
]

from langchain.llms import OpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router import MultiPromptChain
from langchain.chains import ConversationChain

llm = OpenAI(openai_api_key=open_AI_key, temperature=0.5)

destination_chains = {}
for route in routes:
	name = route["name"]
	prompt_template = route["prompt_template"]
	prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
	chain = LLMChain(llm=llm, prompt=prompt)
	destination_chains[name] = chain

financial_prompt = """
Here you will provide the debt ratio of the user found in {input}.
Using the debt ratio, you will summarize the user's financial situation in a {level} tone.
"""

MULTI_PROMPT_ROUTER_TEMPLATE = """\
Given a raw text input to a language model select the model prompt best suited for \
the input. You will be given the names of the available prompts and a description of \
what the prompt is best suited for. You may also revise the original input if you \
think that revising it will ultimately lead to a better response from the language \
model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
	"destination": string \\ name of the prompt to use or "DEFAULT"
	"next_inputs": string \\ a modified version of the original input. It is modified to contai only: the "savings" value, the "debt" value, the "income" value, and the "summary" provided above.
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR \
it can be "DEFAULT" if the input is not well suited for any of the candidate prompts.
REMEMBER: "next_inputs" is not the original input. It is modified to contain: the "savings" value, the "debt" value, the "income" value, and the "summary" provided above.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (must include ```json at the start of the response) >>
<< OUTPUT (must end with ```) >>
"""

prompt = financial_prompt + MULTI_PROMPT_ROUTER_TEMPLATE

destinations = [f"{route['name']}: {route['description']}" for route in routes]
destinations_str = "\n".join(destinations)
router_template = prompt.format(destinations=destinations_str, level=level, input=input)

router_prompt = PromptTemplate(
	template=router_template,
	input_variables=["input"],
	output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(
	router_chain=router_chain,
	destination_chains=destination_chains,
	default_chain=ConversationChain(llm=llm, output_key="text"),
	verbose=False,
)

input = text

# Execute the chain with the input text
output = chain.run(input)

##########################################################################


output_markdown = f"""
## Analysis

**Total Savings:** {output['savings']}
**Monthly Debt:** {output['credit_card_debt']}
**Monthly Income:** {output['income']}

---

## Financial Situation:
{output['financial_situation']}

---

## Recommendation:
{output['recommendation']}
"""

# Use st.markdown() to display the formatted string
st.markdown(output_markdown)
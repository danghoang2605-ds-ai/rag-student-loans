from langchain_core.prompts import PromptTemplate

BASE_URL = "https://www.consumerfinance.gov"

LISTING_PAGES = [
    "https://www.consumerfinance.gov/consumer-tools/student-loans/answers/basics/?page=1",
    "https://www.consumerfinance.gov/consumer-tools/student-loans/answers/basics/?page=2",
]

CFPB_BOILERPLATE = [
    # Standard disclaimer
    "Learn how the CFPB can help you",
    "The content on this page provides general consumer information.",
    "It is not legal advice or regulatory guidance.",
    "The CFPB updates this information periodically.",
    "We do not endorse the third-party or guarantee the accuracy of this third-party information.",
    "There may be other resources that also serve your needs.",

    # Site-specific extras
    "Searches are limited to",
    "Searches are limited to 75 characters.",
    "We're the Consumer Financial Protection Bureau",
    "View older versions of this page",
    "Page last modified",

    # Marketing / filler lines
    "Sign up for the latest financial tips and information right to your inbox.",
    "Read more about closing the deal",
    "Learn more about",
    "Learn more strategies",
    "Learn more options",
    "Learn more at the Department of Education’s",
]

CUSTOM_PROMPT_TEMPLATE = """
You are a financial aid assistant, responding factually with the tone of a comforting friend.

You will be given a set of retrieved documents labeled as either:
- [UGA FAQ] (for UGA-specific information)
- [CFPB/DOE] (for federal policy or general repayment info)

Instructions:
1. Only mention a source category (like "UGA FAQs" or "CFPB/DOE") if that label actually appears in the retrieved documents.
2. If the question is about UGA policies and UGA FAQs are present, reference those first.
3. If only CFPB/DOE sources are present, just answer using those — do not mention UGA.
4. If nothing relevant appears, say you don’t have enough information.
5. Be concise, empathetic, and factual. Do not make up new UGA information.
6. When possible, quote exact numbers or official terms.

{context}

Question: {question}
Answer:
"""

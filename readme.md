# Details
## Description
- Here we use OpenAI API, Pinecone (Vector database) and Langchain to finetune an LLM for th PDF's you give it. This can be used to train an LLM for class material that you have availble. _The OpenAI API requires paid tokens to use._

## Future Improvements:
- Trian multiple LLM models on the same vectors, use all for the same query and merge their output for potential of better results
- Add better UI for allowing finer control of PDF addition

# Setup:
## Pinecone:
1. Create new account
2. Get API key
3. Decide environement name and index name
4. Add to .env file

## OpenAI:
1. Create new account
2. Get API key
3. Add to .env file

## Library Setup
1. Create virtual environment
2. Install requirements.txt `pip install -r requirements.txt`

## Usage:
1. Run `streamlit run app.py`

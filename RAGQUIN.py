import numpy
import os
import glob
from dotenv import load_dotenv
from flask import Flask, jsonify, request
import json 
# Import required libraries
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import openai
from pprint import pprint
from tenacity import retry, wait_random_exponential, stop_after_attempt

from azure.search.documents import SearchClient
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery,
    VectorizableTextQuery
)

from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    AzureOpenAIVectorizer,
    AzureOpenAIParameters,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
    SearchIndex,
    SearchIndexer,
    FieldMapping,
    IndexingParameters,
    IndexingParametersConfiguration,
    BlobIndexerImageAction,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    NativeBlobSoftDeleteDeletionDetectionPolicy,
    SplitSkill,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    AzureOpenAIEmbeddingSkill,
    OcrSkill,
    MergeSkill,
    SearchIndexerIndexProjections,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    IndexProjectionMode,
    SearchIndexerSkillset,
    CognitiveServicesAccountKey
)
from openai import AzureOpenAI
import pyodbc as odbc
# import pyodbc
import pandas as pd
import openai
from dotenv import load_dotenv
load_dotenv()
import openai
from openai import AzureOpenAI
import re
import logging



# Variables not used here do not need to be updated in your .env file
endpoint = os.getenv("endpoint")
credential = AzureKeyCredential(os.getenv("AzureKeyCredential"))
index_name = "msft-test-ocr"
blob_connection_string = os.getenv("blob_connection_string")

# search blob datasource connection string is optional - defaults to blob connection string
# This field is only necessary if you are using MI to connect to the data source
# https://learn.microsoft.com/azure/search/search-howto-indexing-azure-blob-storage#supported-credentials-and-connection-strings
search_blob_connection_string = os.getenv("search_blob_connection_string")
blob_container_name = "document"

azure_openai_type = "azure"
azure_openai_key = os.getenv("azure_openai_key")
azure_openai_endpoint = os.getenv("azure_openai_endpoint")
azure_openai_version = "2024-05-01-preview"
# LLM Model:
azure_openai_llm_deployment_name = "QUERY1"

# Azure OpenAI LLMs and Embedding Models:
# azure_openai_embedding_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
# azure_openai_embedding_key = os.getenv("AZURE_OPENAI_EMBEDDING_KEY")
azure_openai_embedding_version = "2024-02-01"
azure_openai_embedding_deployment = "EMBEDDING"
azure_openai_embedding_model_name = "text-embedding-ada-002"
azure_openai_embedding_model_dimensions = "1536"

def upload_sample_documents(
        blob_connection_string: str,
        blob_container_name: str,
        use_user_identity: bool = True
    ):
    # Connect to Blob Storage
    # Create a BlobServiceClient object using the connection string
    blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
    # Create a container client
    container_client = blob_service_client.get_container_client(blob_container_name)
    if not container_client.exists():
        container_client.create_container()

    documents_directory = os.path.join("input_pdf_documents")
    pdf_files = glob.glob(os.path.join(documents_directory, '*.pdf'))

    for file in pdf_files:
        with open(file, "rb") as data:
            name = os.path.basename(file)
            if not container_client.get_blob_client(name).exists():
                container_client.upload_blob(name=name, data=data)

upload_sample_documents(
    blob_connection_string=blob_connection_string,
    blob_container_name=blob_container_name,
    use_user_identity=True
)
print(f"Setup sample data in blob container `{blob_container_name}` complete ")

indexer_client = SearchIndexerClient(endpoint, credential, api_version="2024-05-01-preview")

container = SearchIndexerDataContainer(name=blob_container_name)
data_source_connection = SearchIndexerDataSourceConnection(
    name=f"{index_name}-blob",
    type="azureblob",
    connection_string=search_blob_connection_string,
    container=container,
    data_deletion_detection_policy=NativeBlobSoftDeleteDeletionDetectionPolicy()
)
data_source = indexer_client.create_or_update_data_source_connection(data_source_connection)

print(f"Data source '{data_source.name}' created or updated")

# Create a search index
# index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
index_client = SearchIndexClient(endpoint=endpoint, credential=credential)


fields = [
    SearchField(name="parent_id", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchField(name="title", type=SearchFieldDataType.String),
    SearchField(name="chunk_id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
    SearchField(name="chunk", type=SearchFieldDataType.String, sortable=False, filterable=False, facetable=False),
    SearchField(name="vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
]

# Configure the vector search configuration

vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(name="myHnsw"),
    ],
    profiles=[
        VectorSearchProfile(
            name="myHnswProfile",
            algorithm_configuration_name="myHnsw",
            vectorizer="myOpenAI",
        )
    ],
    vectorizers=[
        AzureOpenAIVectorizer(
            name="myOpenAI",
            kind="azureOpenAI",
            azure_open_ai_parameters=AzureOpenAIParameters(
                resource_uri=azure_openai_endpoint,
                deployment_id=azure_openai_embedding_deployment,
                api_key=azure_openai_key,
            ),
        ),
    ],
)

# whatever searchable fields we define on top will come here
# # - chunk
semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        content_fields=[SemanticField(field_name="chunk")]
    ),
)

# Create the semantic search with the configuration
semantic_search = SemanticSearch(configurations=[semantic_config])

# Create the search index with the semantic settings
index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_search=semantic_search)
result = index_client.create_or_update_index(index)

# Create a skillset
skillset_name = f"{index_name}-skillset"


split_skill_text_source = "/document/content"
split_skill = SplitSkill(
    description="Split skill to chunk documents",
    text_split_mode="pages",
    context="/document",
    maximum_page_length=2000,
    page_overlap_length=500,
    inputs=[
        InputFieldMappingEntry(name="text", source=split_skill_text_source),
    ],
    outputs=[
        OutputFieldMappingEntry(name="textItems", target_name="pages")
    ],
)

embedding_skill = AzureOpenAIEmbeddingSkill(
    description="Skill to generate embeddings via Azure OpenAI",
    context="/document/pages/*",
    resource_uri=azure_openai_endpoint,
    deployment_id=azure_openai_embedding_deployment,
    api_key=azure_openai_key,
    inputs=[
        InputFieldMappingEntry(name="text", source="/document/pages/*"),
    ],
    outputs=[
        OutputFieldMappingEntry(name="embedding", target_name="vector")
    ],
)

index_projections = SearchIndexerIndexProjections(
    selectors=[
        SearchIndexerIndexProjectionSelector(
            target_index_name=index_name,
            parent_key_field_name="parent_id",
            source_context="/document/pages/*",
            mappings=[
                InputFieldMappingEntry(name="chunk", source="/document/pages/*"),
                InputFieldMappingEntry(name="vector", source="/document/pages/*/vector"),
                InputFieldMappingEntry(name="title", source="/document/metadata_storage_name"),
            ],
        ),
    ],
    parameters=SearchIndexerIndexProjectionsParameters(
        projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS
    ),
)


skills = [split_skill, embedding_skill]

skillset = SearchIndexerSkillset(
    name=skillset_name,
    description="Skillset to chunk documents and generating embeddings",
    skills=skills,
    index_projections=index_projections
)

client = SearchIndexerClient(endpoint, credential)
client.create_or_update_skillset(skillset)
print(f"{skillset.name} created")

# Create an Indexer
indexer_name = f"{index_name}-indexer"

indexer_parameters = None

indexer = SearchIndexer(
    name=indexer_name,
    description="Indexer to index documents and generate embeddings",
    skillset_name=skillset_name,
    target_index_name=index_name,
    data_source_name=data_source.name,
    field_mappings=[FieldMapping(source_field_name="metadata_storage_name", target_field_name="title")],
    parameters=indexer_parameters
)

indexer_client = SearchIndexerClient(endpoint, credential)
indexer_result = indexer_client.create_or_update_indexer(indexer)
indexer_client.run_indexer(indexer_name)

openai.api_type = "azure"
openai.api_base = os.getenv("api_base")
openai.api_version = "2024-05-01-preview"
openai.api_key = os.getenv("api_key")
azure_openai_embedding_deployment = "EMBEDDING"


azure_openai_client = AzureOpenAI(
  api_key = azure_openai_key,
  api_version = "2024-05-01-preview",
  azure_endpoint = azure_openai_endpoint
)

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text):
    return azure_openai_client.embeddings.create(input = [text], model=azure_openai_embedding_deployment).data[0].embedding

def similarity_search(query):
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
    vector_query = VectorizedQuery(vector=generate_embeddings(query), k_nearest_neighbors=3, fields="vector")

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=["chunk" ,"title"],
        query_type=QueryType.SEMANTIC, semantic_configuration_name='my-semantic-config', query_caption=QueryCaptionType.EXTRACTIVE, query_answer=QueryAnswerType.EXTRACTIVE,
        top=3
    )

    results = list(results)

    context = ("\n\n").join([result['chunk'] for result in results])

    return context

def rag_func(query):
    template = """
    You're a helpful assistant.
    Give Response of the Query from the provided context.
    If the answer is not present in the provided context, Reply "Not enough information available." in the Response

    — Start of Context—
    {context}
    — End of Context—
    Query : {question}

    Response:
    """

    context = similarity_search(query)
    prompt = template.format(context=context, question=query)

    response = azure_openai_client.chat.completions.create(
    model = azure_openai_llm_deployment_name,
    messages =[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"{prompt}"}
    ],
    max_tokens=2048,
    temperature=0
    )

    print(response.usage.total_tokens) # check for total tokens
    answer = response.choices[0].message.content
    # print(f"Question: {query}\n")
    # print(f"Answer: {answer}\n")
    # print(f"Context chunks:\n {context}")
    return answer


# from insight_df_summary import insight_df_summary
# from insight import insight
# import streamlit as st
# import o

connection_string = os.getenev("connection_string")
conn = odbc.connect(connection_string)
cursor = conn.cursor()

client = AzureOpenAI(
    azure_endpoint=os.getenv("api_base"),
    api_version="2024-05-01-preview",
    api_key=os.getenv("api_key")
)
print(client)



# Microsoft endpoints:
database = 'ops'
table_name = 'Base_Table'
schema = pd.read_csv("table_schema.csv").to_csv(index=False)



# print(client)

# Get the column descriptions for these multiple tables
def get_column_info(table):
    query = f"""
    SELECT column_name, data_type
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE table_name = '{table}'
    """
    cursor.execute(query)
    columns = cursor.fetchall()
    columns = [(col[0], col[1]) for col in columns]  # Unpack the tuples correctly
    return pd.DataFrame(columns, columns=["column_name", "data_type"])


# Create the SQL Prompt for Query Generation when multiple table flow:
def generate_sql_query_prompt(selected_tables):
    column_info = {}
    for table in selected_tables:
        column_info[table] = get_column_info(table)

    table_descriptions = "SQL tables with their properties:"
    for table, info in column_info.items():
        table_descriptions += f"\n{table} ({', '.join(info['column_name'])})"

    schema = pd.read_csv("table_schema.csv").to_csv(index=False)
    name = pd.read_csv("names.csv").to_csv(index=False)

    # Generate a preview of the data from the table
    top_n_rows_query = f"SELECT TOP 3 * FROM {table_name}"
    sample_data = pd.read_sql(top_n_rows_query, conn)

    instructions_prompt = """
    Read the data description carefully from the tables
    Input Table: """ + table + """
    Sample records are: """ + sample_data.to_csv(index=False) + """
    Schema and column meanings for this table can be understood from: """ + schema + """
    Choose the correct columns and tables that can be used to generate the correct SQL code based on the user question.
    Based on the user question, generate an optimal SQL query. Make sure the column names in the query are as same in the input data.
    If the question is not related to any of these definition, then create SQL query to answer the question. There can be abbreviations in user question, try to understand nearest similar matching term.

    Please provide a valid SQL query to retrieve the required information from the dataset without any errors. The query should run on Azure SQL Database. Ensure the following:
    1. Use descriptive column names enclosed in double quotes (" ").
    2. The query should not use the LIMIT function.
    3. Provide the SQL code block strictly within triple quotes (''') for easy extraction.
    4. Do not name any table "Statistics". Strictly follow this instruction.
    5. Ensure that SQL queries involving division handle divide-by-zero errors using appropriate conditions or `NULLIF` to avoid division by zero.
    6. Try to search for LIKE and not equal to when the filters are generic string values of a column.

    Read the data description carefully from the tables and use the dynamic logic only to choose flags:
    - If 'category' or 'capability' name is mentioned in USER QUERY ,only then use the `Capability_Flag` and 'Capability' column.
    - If the USER QUERY mentions "Category_metrics","metric","metrics", "Channel", or "Attributes" name, use the `Current_Month_AT_Flag` and 'Category_Metrics_Product' column.
    - If colors (e.g., Green, Yellow, Red) are mentioned, prioritize using the relevant flag (`Capability_Flag` or `Current_Month_AT_Flag`) based on the context of the columns identified from the `name` file.

    # Few data related guidelines:
    Consider these unique values for these columnns that are category/capability, category_metrics_product, channel and channel_attribute/attribute: """ + name + """
    1. Always apply filter on the data on Hierarchy_Flag on the basis of USER QUERY. If channel mentioned in USER QUERY then Hierarchy_Flag = 'Channel_Level'. If sub-channel or attribute mentioned in USER QUERY then Hierarchy_Flag = 'Attribute_Level'. If metric or metrics mentioned in USER QUERY then Hierarchy_Flag = 'Metric_Level'.
    2. If 'metric', 'metrics', 'Category_metrics', or 'category metrics product' is mentioned in USER QUERY then consider the 'Category_Metrics_Product' column as 'metric'.
    3. DO NOT use 'Metrics' column unless mentioned explicitly.
    4. For queries related to trends (e.g., metrics going up/down), use the `MOM_Flag` column.
    5. For queries about metrics in a specific color, dynamically choose the correct flag (`Capability_Flag` or `Current_Month_AT_Flag`) based on the user question context and column mappings.

    Exception: If 'Source Lead Completeness' is mentioned in USER QUERY along with a color (green or yellow), then consider the 'Capability_Flag' and 'Executive_cap' column.

    Note: Current_Month_AT_Flag is green means that particular metric or channel or attribute is "performing well" and Current_Month_AT_Flag is yellow or red means that particular metric or channel or attribute is "underperforming".
    When in USER QUERY user asked 'has stated' or 'started performing well' or 'underperforming' or 'have started performing well or underperforming', then you need to consider both Previous_Month_AT_Flag and Current_Month_AT_Flag columns.
    When in USER QUERY user asked has started underperforming or not performing well then you should consider  green in Previous_Month_AT_Flag and red or Yellow in Current_Month_AT_Flag then that means it has started underperforming.
    When in USER QUERY user asked has started or starting to perform well or has improved then it follows two conditions
    Condition 1: Previous Month Flag must be 'Red' with the Current Month Flag being either 'Yellow' or 'Green'.
    Condition 2: Alternatively, the Previous Month Flag can be 'Yellow' with the Current Month Flag as 'Green'.
    When in USER QUERY user aksed "Channel," generate a list of channels categorized under each metric group, organized by each category.



    Use the following sample examples to guide query generation:
    - Q: Which metrics are in yellow for the current month?
      Your approach -> Generate SQL to filter metrics where `Current_Month_AT_Flag = 'Yellow'` and `Month` is the current month.
    - Q: Which metrics are in yellow for the current month, categorized by capability owner?
      Your approach -> Generate SQL to filter metrics where `Current_Month_AT_Flag = 'Yellow'`, group results by `Owner`, and list metrics under each owner.
    - Q: How many metrics have gone up month over month?
      Your approach -> Generate SQL where `MOM_Flag = 'Green'`, indicating metrics with positive MOM changes.
    - Q: Which metrics have gone down month over month?
      Your approach -> Generate SQL where `MOM_Flag = 'Red'`, indicating metrics with negative MOM changes.
    - Q: What color is in ABM recommendation flow?
      Your approach -> Since "ABM recommendation flow" is a capability, use `Capability_Flag` to filter data.
    - Q: What color is in ABM Recommendations Covered By Sellers?
      Your approach -> Since "ABM Recommendations Covered By Sellers" is a metric or Category_Metrics_Product', use `Current_Month_AT_Flag` to filter data.
    - Q: Which metric are performing well?
      Your approach -> Generate SQL to filter attributes where Current_Month_AT_Flag = 'Green'.
    - Q: Which metric are underperforming?
      Your approach -> Generate SQL to filter attributes where Current_Month_AT_Flag IN ('Red', 'Yellow').
    - Q: Which channels were performing well but have started underperforming?
      Your approach -> Generate SQL to filter channels where Previous_Month_AT_Flag = 'Green' and Current_Month_AT_Flag IN ('Red', 'Yellow').
    - Q: Which color does Source Lead Completeness belong to?
      Your approach -> Generate SQL to retrieve `Capability_Flag` where `Executive_cap = 'Source Lead Completeness'`.
    - Q: Which metrics has started performing well or has improved?
      Your approach -> Generate SQL to Filtering Conditions:The Previous Month Flag must be 'Red' with a Current Month Flag of either 'Yellow' or 'Green'Alternatively, the Previous Month Flag can be 'Yellow' with the Current Month Flag being 'Green'.
      Additionly, the Hierarchy Flag must equal 'Metric_Level'.
    - Q: Which channel is underperforming  for Marketo Contact Profile category? 
      Your approach -> Generate SQL to Filtering Conditions: Category = "Marketo Contact Profile category", Hierarchy_Flag = "Channel_Level",Current_Month_AT_Flag IN (''Red','Yellow'),and group by Category_Metrics_Product
    - Q: Which channel is underforming for Completeness Marketo Attributes (Leads) metric?
      Your approach -> Generate SQL to Filtering Conditions : Category_Metrics_Product = "Completeness Marketo Attributes (Leads)",Hierarchy_Flag = "Channel_Level",Current_Month_AT_Flag IN (''Red','Yellow') .

    """

    # sql_gen_prompt = f"{table_descriptions}\n\n{instructions_prompt}"
    sql_gen_prompt = f"\n{instructions_prompt}"

    return sql_gen_prompt

# Use the AzureOpenAI API to generate a SQL query based on the user question and multiple table schema understanding
# (1st LLM call)
def generate_sql_query(user_question, input_tables):
    query_generation_prompt = generate_sql_query_prompt(input_tables) + f"\nQuestion: {user_question}"
    # print(sql_prompt)

    response_completion = client.chat.completions.create(
        model='gpt-4o',
        temperature=0,
        messages=[{'role': 'system', 'content': 'You are a azure SQL query writer.'},
                {"role": "user", "content": query_generation_prompt}]
    )
    output_llm = response_completion.choices[0].message.content

    sqltoken = response_completion.usage.total_tokens

    print(f'sqltoken token {sqltoken}')
    #print(output_llm)
    try:
        sql_query = re.search(r"sql_query'''(.*?)'''", output_llm, re.DOTALL).group(1).strip()

    except:
        sql_query = re.search(r"(sql_query'''|```sql|```)(.*?)('''|```)", output_llm, re.DOTALL).group(2).strip()
    #print(sql_query)

    return sql_query


# Function to generate SQL query based on the provided table name, schema, and user method - API Call 1
def create_sql_query_and_filtered_data(user_question):
    # Get tables for gaming data:
    selected_tables = ['Base_Table']
    # creates the SQL query for execution on Azure DB
    sql_query = generate_sql_query(user_question, selected_tables)
    # print(sql_query)
    # Execute the generated SQL query using the cursor
    query_job = cursor.execute(sql_query)
    # Fetch all the results from the executed query
    data = query_job.fetchall()
    # Convert the fetched results into a list of tuples
    data = [tuple(i) for i in data]
    # Extract column names from the query result
    columns = [x[0] for x in query_job.description]
    # Create a DataFrame from the fetched data and column names
    insight_df = pd.DataFrame(data,columns=columns)
    # Return the generated SQL query and the DataFrame with the query results
    #print("??",insight_df)
    return sql_query, insight_df

# Define a function to generate insights from data using a user-provided business rule method  API Call 2
def insight_generator(input_insight_df, question_prompt):
    insight_prompt  = """
    Generate a concise and clear textual summary to address the following user question, based on the provided insights dataframe.
    Consider the following rules while generating insights:
    - Understand the QUESTION thoroughly and directly address it using the DATA provided.
    - Ensure the summary prioritizes the QUESTION while leveraging the DATA effectively.
    - Include all distinct values from the 'Capability' and 'Metric' columns, regardless of their status change.
    - Make sure to highlight metrics that have shown improvement, specifically those that changed from Red or Yellow to Green, along with any other relevant metrics for a comprehensive overview.
    - If the result is a single record or an absolute value, keep the summary short and brief in 1-2 lines.
    - Provide reasoning for the analysis using relevant data understanding.
    - Highlight key findings, focusing on both improved metrics and those not improved in a structured bullet point format for clarity.
    - Be straightforward in the answer and avoid adding unnecessary information.
    - When in USER QUERY user asked has started or starting to perform well or has improved then it follows two conditions
    Condition 1: Previous Month Flag must be 'Red' with the Current Month Flag being either 'Yellow' or 'Green'.
    Condition 2: Alternatively, the Previous Month Flag can be 'Yellow' with the Current Month Flag as 'Green'.
    - Mention the answer in bullet points
        QUESTION: """+question_prompt+"""
        DATA: """
    
        
    # Request completion from Azure OpenAI based on the prompt
    completion = client.chat.completions.create(
        model= "gpt-4o",
        temperature=0,
        messages=[{'role': 'system', 'content': 'You are a Business Analysis who is expert at Data Interpretation.'},
                {"role": "user", "content": insight_prompt+input_insight_df.to_csv(index=False)}])

    # Extract insights from the response
    insights = completion.choices[0].message.content

    insightstoken = completion.usage.total_tokens
    print(f'insights token {insightstoken}')

    # Return the generated insights
    return insights


# Final response for Chatbot
# print(insight_generator(output_insights_df, user_query))

def quin_func(user_question):
    """
    Wrapper function to integrate SQL query generation, data fetching,
    and insight generation. Returns the generated SQL query, textual
    insights, and the output DataFrame.

    Args:
    - user_question (str): The user's query for generating SQL and insights.

    Returns:
    - generated_sql_query (str): The SQL query generated for the user question.
    - textual_insights (str): The textual insights derived from the data.
    - output_insights_df (pd.DataFrame): DataFrame containing query results.
    """
    try:
        # Step 1: Generate SQL query and fetch filtered data
        generated_sql_query, output_insights_df = create_sql_query_and_filtered_data(user_question)

        # Step 2: Generate textual insights based on the DataFrame and user question
        textual_insights = insight_generator(output_insights_df, user_question)

        # Step 3: Return results
        return generated_sql_query, textual_insights, output_insights_df

    except Exception as e:
        print(f"Error occurred: {e}")
        return None, f"An error occurred: {e}", None

def main_func(user_query):
        prompt = f"""You are a highly intelligent assistant designed to analyze user questions and classify them into two categories:

                RAG (Retrieve-Augment-Generate): These are questions related to definitions, terminology, or detailed explanations about a keyword, topic, or concept.
                SQL: These are questions that involve tables, data, or tasks requiring an SQL query to be generated for a database.
                Your task is to analyze the user's question and respond with exactly one of the following outputs:

                "This question is relevant for RAG."
                "This question is relevant for SQL."
                You must strictly adhere to this classification and provide accurate results. Here are some examples to guide your decision:

                Examples:
                Question: "How do you define Category?"
                Output: "This question is relevant for RAG."

                Question: "Which metric are in yellow for current month."
                Output: "This question is relevant for SQL."

                Question: "What are the objectives of the Operational Scorecard?"
                Output: "This question is relevant for RAG."

                Question: "Which capability are in yellow zone."
                Output: "This question is relevant for SQL."

                When a user provides a question, analyze it carefully and classify it accurately. Provide no extra explanations or text, just the classification output.

                Question: {user_query}
                Output: """


        completion = client.chat.completions.create(
                model= "gpt-4o",
                temperature=0,
                messages=[{'role': 'system', 'content': 'You are a text classifier.'},
                        {"role": "user", "content": prompt}])


        output = completion.choices[0].message.content
        tokens = completion.usage.total_tokens
        print(tokens, output) # comment this out only for

        if "rag" in output.lower():
                rag_answer = rag_func(user_query)
                return rag_answer
        else:
                try:
                        # generated_sql_query, textual_insights, insight_df = main_function(user_query=user_query, selected_tables=selected_tables)
                        generated_sql_query, textual_insights, insight_df = quin_func(user_query)
                        return textual_insights
                except Exception as e:
                        print(f"Error occured {e}")
                        return "Not enough information available."
app = Flask(__name__)

@app.route('/hybridfunction',methods=['POST', 'PUT'])
def main():
    """HTTP trigger for processing queries or setting up resources."""
    
    # Setting up the logging configuration
    logging.info('API Triggered')

    try:
        req_body = request.get_json()
        query = req_body.get("query")
        print(req_body)
        print(query)

        if not query:
            return json.dumps({"error": "Please pass a query in the request."}), 400

        if request.method == "PUT":
            # Handle resource setup
            upload_sample_documents(blob_connection_string, blob_container_name)

            return json.dumps({"success": "Resources set up successfully."}), 200

        elif request.method == "POST":
            answer = main_func(query) 

            response_str = json.dumps(answer)
            return response_str, 200
            
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return json.dumps({"error": f"Error processing request: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)
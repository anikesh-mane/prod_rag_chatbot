## BETTER PROCESSING OF DOCUMENTS 
-   Improve the quality of ingestion pipeline.   


## 2.1 Query Pre-processing & Transformation
Remove llm based "Language Detection." and add intent classification by simple models (ML/BERT)

- Missing Component: Query De-contextualization and Intent Classification.

- Impact: Raw user queries are often poor for vector search. You need a step that rewrites the user's latest message into a search-optimized query based on the conversation history.

- Recommendation: Add a "Query Rewriter" step in the retrieval/ or llm/ module.

## 2.2 Do we need to do vector search for every query?

## 2.3 seperate milvus collection for every document ?
- add document_id to every chunk before ingesting to milvus.
- map conversation_id to document_id in milvus collection.
- when a convo is restarted only search against the document_id of the document that was ingested in that conversation.


## 4.3 Identified Gaps in Session Management
While the "plumbing" for sessions is there, the following logic is missing from both the architecture and the roadmap:

- Session vs. Conversation Mapping: The architecture mentions session:{user_id} in Redis, but a single user may have multiple distinct conversations (e.g., one about "Sales Targets" and another about "IT Support"). You need a session_id or conversation_id that is distinct from the user_id.

- Context Window Management: There is no logic defined for when a session's "history" becomes too large for the LLM's context window. You need a strategy for summarizing or truncating old messages within a session.

- Stateful Metadata: For Channel Sales, a session might need to track "Context Variables" (e.g., which region or product line the user is currently asking about) so the user doesn't have to repeat that information in every query.

## 5.1 Explore RAGAS for evaluation

## 5.2 Guardrails & Safety (Input/Output Moderation)
While the architecture mentions an LLM_CONTENT_FILTERED error code, it doesn't specify the mechanism for enforcement.

- Missing Component: Guardrails Layer (e.g., NeMo Guardrails or Llama Guard).

- Impact: Risks "jailbreaking," PII leakage, or the bot answering questions outside its domain (Channel Sales).

- Recommendation: Implement an input validation step to check for PII/toxicity and an output validation step to ensure the response stays grounded in the retrieved context (Hallucination check).

## 6. Setup Tests and CICD with Github Actions
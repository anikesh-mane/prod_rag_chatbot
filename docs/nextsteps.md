## 1. Conversational Memory (State Management)
The current architecture describes a stateless pipeline: Query → Retrieval → Response. For a chatbot, you need to maintain context across multiple turns.

- Missing Component: A Conversation Buffer or Summary Memory module.

- Impact: Without this, the bot cannot answer follow-up questions (e.g., User: "Who is the CEO?" Bot: "John Doe." User: "How long has he been there?").

- Recommendation: Add a memory/ directory to manage windowed chat history and a logic step to "condense" the chat history into a standalone search query before hitting the retrieval stage.


## 2.1 Query Pre-processing & Transformation
Remove llm based "Language Detection." and add intent classification by simple models (ML/BERT)

- Missing Component: Query De-contextualization and Intent Classification.

- Impact: Raw user queries are often poor for vector search. You need a step that rewrites the user's latest message into a search-optimized query based on the conversation history.

- Recommendation: Add a "Query Rewriter" step in the retrieval/ or llm/ module.

## 2.2 Do we need to do vector search for every query?


## ✅ 3. Document Management & CRUD
The system focuses heavily on the "Query" path, but less on the "Maintenance" path.

- Missing Component: Document Metadata Management API.

- Impact: There is no way for an admin to delete a specific outdated document, update a single chunk, or see what documents are currently indexed without manual DB access.

- Recommendation: Add endpoints in api/routes/admin.py for document status, deletion, and re-indexing.


## 4.1 Implement actual JWT validation
- `api\dependencies.py line 151`

## 4.2 Identified Gaps in Session Management
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
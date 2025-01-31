```mermaid
graph TD;
    A[User Inputs Query] --> B[Retrieve Schema Info - RAG];
    B --> C[Generate SQL Query - LLM];
    C --> D[Execute SQL Query];
    D --> E[Return Query Results];
    C -->|Ambiguous Query?| F[Ask for Clarification];
    F -->|Refined Query| A;

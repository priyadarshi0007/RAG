```mermaid
graph TD;
    A[User Provides DB Connection & Model Selection] --> B[Extract Database Schema];
    B --> C[Store Metadata in Vector Database];
    C --> D[User Inputs Natural Language Query];
    
    D --> E[Retrieve Relevant Schema Details];
    E --> F[Use LLM to Generate SQL Query];
    
    F -->|Valid Query| G[Execute SQL Query in Database];
    F -->|Ambiguous Query| H[Ask User for Clarification];
    
    H --> D;

    G -->|Query Success| I[Format Query Results];
    G -->|Query Fails| J[Handle Errors and Debug];
    
    J -->|Fix and Retry| F;
    I --> K[Return Results to User];

    style A fill:#ffcc00,stroke:#333,stroke-width:2px;
    style F fill:#ff6666,stroke:#333,stroke-width:2px;
    style I fill:#66cc66,stroke:#333,stroke-width:2px;
    style K fill:#66ccff,stroke:#333,stroke-width:2px;

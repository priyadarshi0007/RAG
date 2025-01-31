```mermaid
graph TD;
    A[User Inputs Query] --> B[Retrieve Schema Information];
    B --> C[Check Table Relationships & Structure];
    C --> D[Use LLM to Generate SQL Query];
    
    D -->|Valid Query| E[Execute SQL Query in Database];
    D -->|Ambiguous Query| F[Ask User for Clarification];
    
    F --> A;  

    E -->|Query Executed Successfully| G[Format Query Results];
    E -->|Query Execution Failed| H[Handle Errors & Debug];
    
    H -->|Fix Issues| D;  
    G --> I[Return Results to User];

    style A fill:#ffcc00,stroke:#333,stroke-width:2px;
    style B fill:#ffcc00,stroke:#333,stroke-width:2px;
    style C fill:#ffcc00,stroke:#333,stroke-width:2px;
    style D fill:#ff6666,stroke:#333,stroke-width:2px;
    style E fill:#ff6666,stroke:#333,stroke-width:2px;
    style F fill:#ff9966,stroke:#333,stroke-width:2px;
    style G fill:#66cc66,stroke:#333,stroke-width:2px;
    style H fill:#ff6666,stroke:#333,stroke-width:2px;
    style I fill:#66ccff,stroke:#333,stroke-width:2px;

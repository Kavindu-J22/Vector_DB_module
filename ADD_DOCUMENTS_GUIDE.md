# 📚 **HOW TO ADD MORE LEGAL DOCUMENTS**

## 🚀 **Quick Start - Add Documents Now!**

### **Method 1: Run Expanded Demo (Instant - 66 Documents)**
```bash
python expanded_legal_documents.py
```
**This adds 66 comprehensive legal documents across 10 legal areas!**

### **Method 2: Add Custom Documents to Existing Demo**

Edit `working_demo.py` and add your documents to the `create_sample_documents()` function:

```python
def create_sample_documents(self) -> List[Dict[str, Any]]:
    documents = [
        # Existing documents...
        
        # ADD YOUR NEW DOCUMENTS HERE:
        {
            "id": "your_doc_001",
            "title": "Your Document Title",
            "content": "Your legal document content here...",
            "doctrine": "contract_law",  # or tort_law, constitutional_law, etc.
            "court": "district_court",
            "year": 2023,
            "jurisdiction": "federal"
        },
        # Add more documents...
    ]
    return documents
```

## 📋 **Document Structure Template**

### **Required Fields:**
```python
{
    "id": "unique_document_id",           # Unique identifier
    "title": "Document Title",            # Descriptive title
    "content": "Full document text...",   # Main legal content
    "doctrine": "legal_doctrine",         # Legal area
    "court": "court_level",              # Court type
    "year": 2023,                       # Year (integer)
    "jurisdiction": "federal_or_state"   # Jurisdiction
}
```

### **Legal Doctrine Options:**
- `contract_law` - Contract disputes, agreements, breach
- `tort_law` - Personal injury, negligence, liability
- `constitutional_law` - Constitutional rights, amendments
- `criminal_law` - Criminal procedure, evidence, rights
- `property_law` - Real estate, ownership, possession
- `corporate_law` - Business law, shareholder disputes
- `family_law` - Divorce, custody, domestic relations
- `employment_law` - Workplace rights, discrimination
- `intellectual_property` - Patents, trademarks, copyright
- `environmental_law` - Environmental regulations, compliance

### **Court Level Options:**
- `supreme_court` - Highest court decisions
- `appellate_court` - Appeals court cases
- `federal_court` - Federal district courts
- `district_court` - State district courts
- `superior_court` - State superior courts
- `family_court` - Family law matters
- `commercial_court` - Business disputes
- `chancery_court` - Equity matters

## 🎯 **Examples of Documents You Can Add**

### **Contract Law Example:**
```python
{
    "id": "contract_new_001",
    "title": "Software License Breach - SaaS Agreement",
    "content": """
    TechStart Inc. licensed cloud-based accounting software from CloudSoft under a 
    three-year Software as a Service (SaaS) agreement. The license included unlimited 
    users for $10,000 monthly with data storage up to 1TB. TechStart exceeded the 
    storage limit by 500GB for six months without upgrading their plan. CloudSoft 
    terminated the agreement and seeks damages for breach of contract terms. The case 
    involves interpretation of SaaS licensing terms, usage limitations, and remedies 
    for contract violations in cloud computing agreements.
    """,
    "doctrine": "contract_law",
    "court": "commercial_court",
    "year": 2023,
    "jurisdiction": "federal"
}
```

### **Tort Law Example:**
```python
{
    "id": "tort_new_001", 
    "title": "Autonomous Vehicle Accident - Product Liability",
    "content": """
    Plaintiff was injured when an autonomous vehicle manufactured by AutoTech Corp 
    failed to detect a pedestrian crossing and struck the plaintiff at a crosswalk. 
    The vehicle's AI system was operating in full autonomous mode when the accident 
    occurred. Investigation revealed that the vehicle's sensors were functioning 
    properly, but the AI decision-making algorithm failed to properly classify the 
    pedestrian as an obstacle requiring emergency braking. This case presents novel 
    questions about product liability for AI-powered systems and the standard of 
    care for autonomous vehicle manufacturers.
    """,
    "doctrine": "tort_law",
    "court": "federal_court", 
    "year": 2023,
    "jurisdiction": "federal"
}
```

### **Constitutional Law Example:**
```python
{
    "id": "constitutional_new_001",
    "title": "Digital Privacy Rights - Fourth Amendment Online",
    "content": """
    Law enforcement obtained defendant's location data from his smartphone without 
    a warrant by requesting the information directly from the cellular service 
    provider. The data revealed defendant's movements over a 30-day period and was 
    used to establish his presence at multiple crime scenes. Defendant argues that 
    warrantless collection of digital location data violates his Fourth Amendment 
    rights and that the digital age requires enhanced privacy protections. The case 
    addresses the intersection of technology and constitutional privacy rights in 
    the digital era.
    """,
    "doctrine": "constitutional_law",
    "court": "supreme_court",
    "year": 2023,
    "jurisdiction": "federal"
}
```

## 🔧 **How to Add Documents Programmatically**

### **Method 3: Create Your Own Document Loader**

```python
def add_custom_documents():
    """Add your custom legal documents."""
    
    # Initialize vector database
    vector_db = VectorDatabase()
    
    # Your custom documents
    custom_documents = [
        {
            "id": "custom_001",
            "title": "Your Case Title",
            "content": "Your legal document content...",
            "doctrine": "contract_law",
            "court": "district_court", 
            "year": 2023,
            "jurisdiction": "state"
        },
        # Add more documents here...
    ]
    
    # Add documents to database
    success = vector_db.add_documents(custom_documents)
    
    if success:
        print(f"✅ Added {len(custom_documents)} custom documents")
        
        # Test search
        results = vector_db.search("your search query", top_k=5)
        for result in results:
            print(f"Found: {result['title']}")
    else:
        print("❌ Failed to add documents")

# Run your custom loader
add_custom_documents()
```

## 📊 **Current Document Collection**

### **Expanded Collection (66 Documents):**
- **Contract Law**: 5 documents
- **Tort Law**: 3 documents  
- **Constitutional Law**: 2 documents
- **Criminal Law**: 1 document
- **Property Law**: 1 document
- **Corporate Law**: 1 document
- **Family Law**: 1 document
- **Employment Law**: 1 document
- **Intellectual Property**: 1 document
- **Environmental Law**: 1 document

### **Original Collection (5 Documents):**
- Contract Law Principles
- Tort Liability Standards
- Constitutional Rights Framework
- Criminal Procedure Guidelines
- Property Ownership Rules

## 🎯 **Best Practices for Adding Documents**

### **Content Quality:**
- ✅ Use realistic legal scenarios
- ✅ Include specific legal terminology
- ✅ Reference relevant laws and precedents
- ✅ Provide sufficient detail for meaningful search

### **Metadata Accuracy:**
- ✅ Choose appropriate legal doctrine
- ✅ Select correct court level
- ✅ Use realistic years (2019-2023)
- ✅ Specify proper jurisdiction

### **Document Length:**
- ✅ Minimum: 200 words for meaningful content
- ✅ Optimal: 300-500 words for rich context
- ✅ Maximum: 1000 words to avoid processing issues

## 🚀 **Testing Your New Documents**

### **After Adding Documents:**

1. **Run the demo:**
   ```bash
   python working_demo.py
   ```

2. **Test specific searches:**
   ```bash
   python accuracy_evaluation.py
   # Choose option 2 for manual testing
   ```

3. **Search for your content:**
   - Use keywords from your documents
   - Test different legal terminology
   - Verify correct classification

### **Verification Checklist:**
- [ ] Documents load without errors
- [ ] Search finds your documents
- [ ] Classification is accurate
- [ ] Relevance scores are reasonable
- [ ] Metadata filtering works

## 📈 **Scaling to Large Collections**

### **For 100+ Documents:**
```python
# Use batch processing
def add_large_document_collection(documents):
    vector_db = VectorDatabase()
    
    # Process in batches of 50
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        success = vector_db.add_documents(batch)
        print(f"Processed batch {i//batch_size + 1}: {success}")
```

### **For 1000+ Documents:**
- Use the Pinecone cloud integration
- Enable performance optimization
- Consider distributed processing

## 🎉 **Quick Demo Commands**

### **See Expanded Collection:**
```bash
python expanded_legal_documents.py
```

### **Add Your Own Documents:**
1. Edit `working_demo.py`
2. Add documents to `create_sample_documents()`
3. Run: `python working_demo.py`

### **Test Search Quality:**
```bash
python accuracy_evaluation.py
```

**Your Vector Database Module can handle unlimited legal documents! Start adding your collection now!** 🚀

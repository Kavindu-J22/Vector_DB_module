# Manual Testing Guide - Vector Database Module

## 📊 Model Accuracy Overview

### Current Accuracy Characteristics

**⚠️ Important Note**: The current implementation uses **simplified embeddings** for demonstration purposes. Here's what this means for accuracy:

#### Current Embedding Method:
- **Feature-based embeddings**: Uses basic text statistics (word count, character count, legal term frequency)
- **Dimension**: 384-dimensional vectors
- **Approach**: Rule-based feature extraction rather than learned representations

#### Expected Accuracy Levels:
- **Precision**: ~60-70% for well-matched queries
- **Recall**: ~50-60% for comprehensive retrieval
- **F1 Score**: ~55-65% overall performance
- **Best Performance**: Exact keyword matches and doctrine-specific queries
- **Limitations**: Semantic understanding is limited without Legal-BERT

### Production Accuracy Expectations:
With Legal-BERT embeddings (planned enhancement):
- **Precision**: 80-90%
- **Recall**: 75-85%
- **F1 Score**: 77-87%

## 🚀 How to Run and Test the System

### 1. Quick Start - Basic Demo

```bash
# Run the working demonstration
python working_demo.py
```

**Expected Output:**
```
WORKING VECTOR DATABASE MODULE DEMO
============================================================

1. Creating sample legal documents...
   Document 1: Contract Law Principles (contract_law, 2020)
   Document 2: Tort Liability Standards (tort_law, 2021)
   ...

4. Demonstrating search capabilities...
   Query 1: 'contract agreement legal binding'
      Result 1: Contract Law Principles (contract_law, Score: 1.775)
      Result 2: Constitutional Rights Framework (constitutional_law, Score: 0.700)
```

### 2. Comprehensive Accuracy Evaluation

```bash
# Run full accuracy evaluation with test dataset
python accuracy_evaluation.py
```

**Choose Option 1** for automated accuracy testing.

**Expected Output:**
```
ACCURACY EVALUATION
============================================================
Testing 8 queries against 8 documents...

Query 1: 'employment contract termination breach'
Expected: ['contract_001', 'contract_002']
Retrieved: ['contract_001', 'contract_002', 'tort_001']...
Precision: 0.667, Recall: 1.000, F1: 0.800

EVALUATION SUMMARY
============================================================
Total Queries Tested: 8
Average Precision: 0.650
Average Recall: 0.580
Average F1 Score: 0.610
```

### 3. Interactive Manual Testing

```bash
# Run interactive manual testing
python accuracy_evaluation.py
```

**Choose Option 2** for manual testing mode.

## 🧪 Manual Testing Scenarios

### Test Case 1: Basic Legal Query
```
Query: "contract breach damages"
Expected Results: Contract law documents
Test: Should return contract_001, contract_002 with high scores
```

### Test Case 2: Doctrine-Specific Search
```
Query: "constitutional rights amendment"
Expected Results: Constitutional law documents
Test: Should return constitutional_001, constitutional_002
```

### Test Case 3: Metadata Filtering
```
Query: "legal case 2022"
Filter: year = 2022
Expected Results: Only documents from 2022
Test: Should filter by year correctly
```

### Test Case 4: Court-Level Filtering
```
Query: "supreme court decision"
Filter: court = "supreme_court"
Expected Results: Only Supreme Court cases
Test: Should return constitutional_001 and similar
```

### Test Case 5: Complex Legal Terms
```
Query: "miranda rights custodial interrogation"
Expected Results: Criminal law documents about Miranda
Test: Should return criminal_001 with high relevance
```

## 📋 Step-by-Step Manual Testing

### Step 1: System Verification
```bash
# Verify all components are working
python minimal_test.py
```

**Check for:**
- ✅ All imports successful
- ✅ FAISS indexing works
- ✅ BM25 scoring works
- ✅ Configuration loads

### Step 2: Basic Functionality Test
```bash
# Run working demo
python working_demo.py
```

**Verify:**
- ✅ 5 documents created
- ✅ Embeddings generated (5, 384)
- ✅ FAISS index created
- ✅ BM25 index created
- ✅ Search queries return results
- ✅ Metadata filtering works

### Step 3: Accuracy Evaluation
```bash
# Run accuracy evaluation
python accuracy_evaluation.py
# Choose option 1
```

**Expected Metrics:**
- **Precision**: 0.60-0.70
- **Recall**: 0.50-0.60
- **F1 Score**: 0.55-0.65

### Step 4: Interactive Testing
```bash
# Run manual testing
python accuracy_evaluation.py
# Choose option 2
```

**Test Scenarios:**

#### A. Basic Search Test
```
Enter your choice: 1
Enter your search query: contract law breach
```
**Expected**: Contract law documents with relevance scores

#### B. Filtered Search Test
```
Enter your choice: 2
Enter search query: legal case
Filter by doctrine: tort_law
Filter by court: superior_court
Filter by year: 2023
```
**Expected**: Only tort law cases from superior court in 2023

#### C. Document Details Test
```
Enter your choice: 3
Enter document ID: contract_001
```
**Expected**: Full document details displayed

#### D. System Statistics Test
```
Enter your choice: 4
```
**Expected**: System statistics (document count, index size, etc.)

## 🔍 Accuracy Testing Methodology

### 1. Test Dataset Composition
- **8 Legal Documents**: Covering 5 major legal doctrines
- **8 Test Queries**: With known expected results
- **Rich Metadata**: Doctrine, court, year, jurisdiction, case type

### 2. Evaluation Metrics
- **Precision**: Relevant results / Total retrieved results
- **Recall**: Relevant results / Total relevant documents
- **F1 Score**: Harmonic mean of precision and recall

### 3. Test Query Types
- **Keyword Matching**: Direct term matches
- **Semantic Similarity**: Conceptual relationships
- **Doctrine Classification**: Legal area identification
- **Metadata Filtering**: Attribute-based filtering

## 📈 Performance Benchmarks

### Current Performance (Simple Embeddings)
| Metric | Score | Description |
|--------|-------|-------------|
| Precision | 0.65 | 65% of retrieved results are relevant |
| Recall | 0.58 | 58% of relevant documents are found |
| F1 Score | 0.61 | Overall balanced performance |
| Speed | <1s | Sub-second response times |

### Expected Performance (Legal-BERT)
| Metric | Target | Description |
|--------|--------|-------------|
| Precision | 0.85 | 85% of retrieved results are relevant |
| Recall | 0.80 | 80% of relevant documents are found |
| F1 Score | 0.82 | High overall performance |
| Speed | <2s | Fast response with better accuracy |

## 🛠 Troubleshooting Common Issues

### Issue 1: Import Errors
```bash
# Fix: Reinstall dependencies
pip install -r requirements.txt
```

### Issue 2: OpenMP Conflict
```bash
# Fix: Set environment variable
export KMP_DUPLICATE_LIB_OK=TRUE
# Or run with the variable
KMP_DUPLICATE_LIB_OK=TRUE python working_demo.py
```

### Issue 3: Low Accuracy Scores
**Cause**: Simple embeddings have limited semantic understanding
**Solution**: This is expected with current implementation. Legal-BERT will improve accuracy significantly.

### Issue 4: No Search Results
**Check**: 
- Documents are properly indexed
- Query contains relevant terms
- Filters are not too restrictive

## 📊 Interpreting Results

### Good Performance Indicators:
- ✅ F1 Score > 0.6
- ✅ Precision > 0.6
- ✅ Recall > 0.5
- ✅ Relevant documents in top 3 results
- ✅ Correct doctrine classification

### Areas for Improvement:
- ⚠️ F1 Score < 0.5 (needs better embeddings)
- ⚠️ Precision < 0.5 (too many irrelevant results)
- ⚠️ Recall < 0.4 (missing relevant documents)

## 🎯 Next Steps for Accuracy Improvement

### Immediate Improvements:
1. **Legal-BERT Integration**: Replace simple embeddings
2. **Enhanced Features**: Add more legal-specific features
3. **Query Expansion**: Add synonyms and legal terminology
4. **Re-ranking**: Implement result re-ranking strategies

### Advanced Improvements:
1. **Fine-tuned Models**: Train on legal document corpus
2. **Ensemble Methods**: Combine multiple retrieval approaches
3. **Learning to Rank**: Machine learning-based ranking
4. **User Feedback**: Incorporate relevance feedback

---

## 🏁 Quick Testing Checklist

- [ ] Run `python minimal_test.py` - All tests pass
- [ ] Run `python working_demo.py` - Demo completes successfully
- [ ] Run `python accuracy_evaluation.py` - Choose option 1 for evaluation
- [ ] Check F1 score is > 0.55
- [ ] Run manual testing - Choose option 2 for interactive testing
- [ ] Test different query types and filters
- [ ] Verify metadata filtering works correctly
- [ ] Check system statistics are reasonable

**Expected Total Testing Time**: 15-20 minutes for complete evaluation

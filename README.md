# Multi-level Knowledge Graph Neural Network Explains Assays for Breast Cancer Recurrence

# Abstract
Motivation: Multi-gene assays have been widely used to predict the risk of recurrence for hormone receptor(HR)-positive breast cancer with genes that are closely associated with survival or recurrence outcomes. However, due to the design goal of outcome-driven approaches, the assays lack the explanatory power on why cancer recurrence risk is high or low given the transcriptional quantities of the assays’ genes.
Results: To improve the explanatory power of the assays, we developed a multi-level knowledge graph neural network model that explains regulatory pathways of assay genes using an attention mechanism. From a multi-gene assay and biological pathway database, a multi-level knowledge graph is constructed and organized into transcriptional subpathways involving cancer driver genes and assay genes through a subpathway cascade to reflect the multi-step nature of breast cancer recurrence. After construction of the regulatory landscape, an attention-based graph neural network is trained and used to predict the recurrence risk of breast cancer. We evaluated our model for three multi-gene assays, Oncotype DX, EndoPredict, and Prosigna, on breast cancer patients in SCAN-B dataset. Our approach improved the predictive powers of assays for breast cancer recurrence and, more importantly, explained the recurrence risk in terms of regulatory mechanisms. By interpreting the regulatory landscape with attention weights, we found that all three assays are mainly regulated by signaling pathways driving cancer proliferation, especially Ras and Estrogen signaling pathways, for breast cancer recurrence.

# Implementation
## Subpathway Cascade Method Schema
 
We propose an assay-agnostic knowledge graph generation method called the subpathway cascade, which identifies potential subpathways that regulate transcriptomic states of assay genes and are involved in the process of breast cancer recurrence. Generating all regulatory subpathways connecting cancer driver genes to assay genes including cascading cases is computationally expensive due to the numerous combinations of regulatory mechanisms. To address this challenge, we propose a two-step generating process. To implement the subpathwy cascade, AssayGeneSetName should be in [UpdatedEndoPredictCancer, UpdatedOncotypeDXCancer, UpdatedProsigna], corresponding to EndoPredict, OncotypeDX and Prosigna, respectively. This version currently supports three sets of assay genes. The size of the file required to execute the method is too large, so only the code and results for the result are uploaded. Results for each assay gene set exist in the SubpathwayCascade folder.

### Step1 : Generating cascading backbones
Code file : SubpathwayCascade-1.CascadeBackbones.py
```ShellSession
$ python SubpathwayCascade-1.CascadeBackbones.py --TARGET_GENE_SET [AssayGeneSetName]
```

To generate potential regulatory subpathways of assay genes, we generate cascading backbones which contain essential information about the subpathways in the form of two sequences. The cascading backbones identified from the algorithm are stored in CascadeBackbones.txt in the corresponding folder.

### Step2 : Assay-specific subpathway graphs
Code file : SubpathwayCascade-2.KnowledgeGraphGeneration.py
```ShellSession
$ python SubpathwayCascade-2.KnowledgeGraphGeneration.py --TARGET_GENE_SET [AssayGeneSetName]
```

This code creates a multi-level knowledge graph in the form of a torch object from the cascading backbones created in the previous step. The multi-level knowledge graph neural network identifies regulatory subpathways important for breast cancer recurrence as an attention mechanism using the generated graph structure. In order to construct a knowledge graph that can be used for a specific dataset, a process of mapping to the genes included in the dataset is required. The code file for the SCAN-B dataset is SubpathwayCascade-2.KnowledgeGraphGeneration.MappingforSCAN-B.py.

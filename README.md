# Multi-level Knowledge Graph Neural Network Explains Assays for Breast Cancer Recurrence

[ExplainableMLKGNN.pdf](https://github.com/JoonHyeongPark/ExplainableMLKGNN/files/10554431/ExplainableMLKGNN.pdf)

[ExplainableMLKGNN-SupplementaryMaterials.pdf](https://github.com/JoonHyeongPark/ExplainableMLKGNN/files/10554430/ExplainableMLKGNN-SupplementaryMaterials.pdf)

# Abstract
Motivation: Multi-gene assays have been widely used to predict the risk of recurrence for hormone receptor(HR)-positive breast cancer with genes that are closely associated with survival or recurrence outcomes. However, due to the design goal of outcome-driven approaches, the assays lack the explanatory power on why cancer recurrence risk is high or low given the transcriptional quantities of the assays’ genes.
Results: To improve the explanatory power of the assays, we developed a multi-level knowledge graph neural network model that explains regulatory pathways of assay genes using an attention mechanism. From a multi-gene assay and biological pathway database, a multi-level knowledge graph is constructed and organized into transcriptional subpathways involving cancer driver genes and assay genes through a subpathway cascade to reflect the multi-step nature of breast cancer recurrence. After construction of the regulatory landscape, an attention-based graph neural network is trained and used to predict the recurrence risk of breast cancer. We evaluated our model for three multi-gene assays, Oncotype DX, EndoPredict, and Prosigna, on breast cancer patients in SCAN-B dataset. Our approach improved the predictive powers of assays for breast cancer recurrence and, more importantly, explained the recurrence risk in terms of regulatory mechanisms. By interpreting the regulatory landscape with attention weights, we found that all three assays are mainly regulated by signaling pathways driving cancer proliferation, especially Ras and Estrogen signaling pathways, for breast cancer recurrence.

# Tutorial
### Step1 : Generating cascading backbones
SubpathwayCascade-1.CascadeBackbones.py

### Step2 : Assay-specific subpathway graphs
SubpathwayCascade-2.KnowledgeGraphGeneration.py

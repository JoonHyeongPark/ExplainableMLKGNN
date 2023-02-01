import argparse
import random

from itertools import chain, product
from collections import defaultdict
from tqdm import tqdm
from pandas.errors import EmptyDataError

import itertools
import pandas as pd
import os
import networkx as nx
import pickle

import sys
sys.setrecursionlimit(10 ** 9)


KEGG_SIGNALING_PATHWAYS = ["04010","04012","04014","04015","04310","04330","04340","04350","04390","04370",
                           "04371","04630","04064","04668","04066","04068","04020","04072","04071","04024",
                           "04022","04151","04152","04150","04115","04550","04620","04621","04622","04625","04660",
                           "04657","04662","04664","04062","04910","04922","04920","03320","04912","04915",
                           "04917","04921","04926","04919","04722"]
KEGG_SIGNALING_PATHWAYS = ["hsa" + id_number for id_number in KEGG_SIGNALING_PATHWAYS]


def argument_parsing() :

    parser = argparse.ArgumentParser(description='SubpathwayCascade')

    parser.add_argument('--SOURCE_GENE_SET', type=str, help='source gene set name', default='CGC_Hallmark_BreastCancer')
    parser.add_argument('--TARGET_GENE_SET', type=str, help='target gene set name', default='UpdatedOncotypeDXCancer')
    
    parser.add_argument('--BLOCK_CUTOFF', type=int, help='cascade block cutoff', default=2)
    parser.add_argument('--BRIDGE_CUTOFF', type=int, help='bridge cutoff', default=1)
    parser.add_argument('--EVIDENCE_CUTOFF', type=int, help='TFTG evidence cutoff', default=5)
    
    args = parser.parse_args()
    
    return args

def read_source_target_gene_sets() :
    
    tier1_CGC = pd.read_csv(os.path.join(os.getcwd(), "data", "COSMIC", "COSMIC_GRCh38_COSMICv96_Tier1.tsv"), sep="\t")
    tier1_CGC_hallmark = tier1_CGC[tier1_CGC["Hallmark"].notna()]
    tier1_CGC_hallmark_genes = set(tier1_CGC_hallmark["Gene Symbol"])
            
    hallmarks = pd.read_csv(os.path.join(os.getcwd(), "data", "COSMIC", "Cancer_Gene_Census_Hallmarks_Of_Cancer.tsv"), sep="\t", encoding="unicode_escape")
    total_hallmark_genes = set(hallmarks[hallmarks["HALLMARK"] != "mouse model"]["GENE_NAME"])
    breast_cancer_hallmark_genes = set(hallmarks[(hallmarks["CELL_TYPE"].str.contains("breast")) & (hallmarks["HALLMARK"] != "mouse model")]["GENE_NAME"])
    
    assay_genes_folder = os.path.join(os.getcwd(), "data", "AssayGenes")
    
    pam50_genes = set(pd.read_csv(os.path.join(assay_genes_folder, "pam50.tsv"), sep="\t")["probe"])
    endopredict_genes = set(pd.read_csv(os.path.join(assay_genes_folder, "sig.endopredict.tsv"), sep="\t")["symbol"])
    oncotypedx_genes = set(pd.read_csv(os.path.join(assay_genes_folder, "sig.oncotypedx.tsv"), sep="\t")["symbol"])
    
    oncofree_genes = set("ACTB, APOBEC3B, ASF1B, ASPM, AURKA, AURKB, BAG1, BCL2, BIRC5, BLM, BUB1, BUB1B, C14orf45, C16orf61, C7orf63, CACNA1D, CCNA2, CCNB1, CCNB2, CCNE1, CCNE2, CCT5, CD68, CDC20, CDC25A, CDC45, CDC6, CDCA3, CDCA8, CDK1, CDKN3, CENPA, CENPE, CENPF, CENPM, CENPN, CEP55, CHEK1, CIRBP, CKS2, CRIM1, CTSL2, CX3CR1, CYBRD1, DBF4, DDX39, DLGAP5, DNMT3B, DONSON, DTL, E2F1, E2F8, ECHDC2, ERBB2, ERCC6L, ESPL1, ESR1, EXO1, EZH2, FAM64A, FANCI, FBXO5, FEN1, FOXM1, GAPDH, GINS1, GRB7, GSTM1, GTSE1, GUSB, HJURP, HMMR, HN1, IFT46, KIF11, KIF13B, KIF14, KIF15, KIF18A, KIF18B, KIF20A, KIF23, KIF2C, KIF4A, KIFC1, KPNA2, LMNB1, LMNB2, LRIG1, LRRC48, LRRC59, MAD2L1, MARCH8, MCM10, MCM2, MCM6, MELK, MKI67, MLF1IP, MMP11, MYBL2, NCAPG, NCAPG2, NCAPH, NDC80, NEK2, NUP93, NUSAP1, OIP5, PBK, PDSS1, PGR, PKMYT1, PLK1, PLK4, PRC1, PTTG1, RACGAP1, RAD51, RAD51AP1, RAI2, RFC4, RPLP0, RRM2, SCUBE2, SETBP, SF3B3, SHCBP1, SHMT2, SLC25A1, SLC7A5, SPAG5, SPC25, SQLE, STARD13, STIL, STMN1, SYNC, TACC3, TFRC, TK1, TOP2A, TPX2, TRIP13, TROAP, TTK, UBE2C, UBE2S, ZWINT".split(", "))
    
    def extract_updated_symbols(HUGO_dataframe, target_symbols) :

        preserved_symbol = HUGO_dataframe[HUGO_dataframe["Match type"] == "Approved symbol"]
        updated_symbol = HUGO_dataframe[HUGO_dataframe["Match type"] == "Previous symbol"]

        to_updated_symbol = dict(zip(target_symbols, target_symbols))
        to_updated_symbol.update(dict(zip(updated_symbol["Input"], updated_symbol["Approved symbol"])))
        to_updated_symbol.update(dict(zip(preserved_symbol["Input"], preserved_symbol["Approved symbol"])))

        return set([to_updated_symbol[symbol] for symbol in target_symbols])

    PAM50_SYMBOLS = extract_updated_symbols(pd.read_csv(os.path.join(assay_genes_folder, "sig.pam50.HUGO.tsv"), sep="\t"), pam50_genes)
    ENDOPREDICT_SYMBOLS = extract_updated_symbols(pd.read_csv(os.path.join(assay_genes_folder, "sig.endopredict.HUGO.tsv"), sep="\t"), endopredict_genes)
    ONCOTYPEDX_SYMBOLS = extract_updated_symbols(pd.read_csv(os.path.join(assay_genes_folder, "sig.oncotypedx.HUGO.tsv"), sep="\t"), oncotypedx_genes)

    if SOURCE_GENE_SET == "CGC_Hallmark" : SOURCE_GENES = total_hallmark_genes
    elif SOURCE_GENE_SET == "CGC_Hallmark_Tier1" : SOURCE_GENES = tier1_CGC_hallmark_genes
    elif SOURCE_GENE_SET == "CGC_Hallmark_BreastCancer" : SOURCE_GENES = breast_cancer_hallmark_genes

    if TARGET_GENE_SET == "OncotypeDX" :
        TARGET_GENES = oncotypedx_genes
    elif TARGET_GENE_SET == "OncoFREE" :
        TARGET_GENES = oncofree_genes
    elif TARGET_GENE_SET == "OncotypeDXCancer" :
        TARGET_GENES = set(["BCL2" if gene in ["ACTB", "GAPDH", "RPLP0", "GUSB", "TFRC"] else gene for gene in oncotypedx_genes]) 
    elif TARGET_GENE_SET == "EndoPredict" :
        TARGET_GENES = endopredict_genes
    elif TARGET_GENE_SET == "EndoPredictCancer" :
        TARGET_GENES = set("BIRC5, UBE2C, DHCR7, RBBP8, IL6ST, AZGP1, MGP, STC2".split(", "))
    elif TARGET_GENE_SET == "Prosigna" :
        TARGET_GENES = set("FOXC1 MIA NDC80 CEP55 ANLN MELK GPR160 TMEM45B ESR1 FOXA1 ERBB2 GRB7 FGFR4 BLVRA BAG1 CDC20 CCNE1 ACTR3B MYC SFRP1 KRT14 KRT5 MLPH CCNB1 CDC6 TYMS UBE2T RRM2 MMP11 CXXC5 ORC6 MDM2 KIF2C PGR MKI67 BCL2 EGFR PHGDH CDH3 CDH15 NAT1 SLC39A6 MAPT UBE2C PTTG1 EXO1 CENPF NUF2 MYBL2 BIRC5".split())
    elif TARGET_GENE_SET == "UpdatedOncotypeDX" :
        TARGET_GENES = ONCOTYPEDX_SYMBOLS
    elif TARGET_GENE_SET == "UpdatedEndoPredict" :
        TARGET_GENES = ENDOPREDICT_SYMBOLS
    elif TARGET_GENE_SET == "UpdatedProsigna" :
        TARGET_GENES = PAM50_SYMBOLS
    elif TARGET_GENE_SET == "UpdatedOncotypeDXCancer" :
        TARGET_GENES = ONCOTYPEDX_SYMBOLS - set(["ACTB", "GAPDH", "RPLP0", "GUSB", "TFRC"])
    elif TARGET_GENE_SET == "UpdatedEndoPredictCancer" :
        TARGET_GENES = set("BIRC5, UBE2C, DHCR7, RBBP8, IL6ST, AZGP1, MGP, STC2".split(", "))
    
    return SOURCE_GENES, TARGET_GENES
    
def read_external_TFTG_reference(evidence_cutoff=2) :
    
    TFTG_path = os.path.join(os.getcwd(), "data", "TFLink")
        
    TFLink_All_file = "TFLink_Homo_sapiens_interactions_All_simpleFormat_v1.0.tsv"
    TFLink_All_df = pd.read_csv(os.path.join(TFTG_path, TFLink_All_file), sep="\t")
    
    def filter_TFLink(TFLink_All_df, small_scale, evidence_cutoff) :

        if small_scale :
            TFLink_Filtered_df = TFLink_All_df[(TFLink_All_df["PubmedID"].str.count(";") + 1 >= evidence_cutoff) & (TFLink_All_df["Small-scale.evidence"] == "Yes")]
        else :
            TFLink_Filtered_df = TFLink_All_df[(TFLink_All_df["PubmedID"].str.count(";") + 1 >= evidence_cutoff) | (TFLink_All_df["Small-scale.evidence"] == "Yes")]

        TFLink_Filtered_dict = defaultdict(set)
        TFLink_Filtered_TFTGs = list()

        TG_TF_graph = nx.DiGraph()
        for TF, TG in TFLink_Filtered_df[["Name.TF", "Name.Target"]].to_numpy() :
            if ";" in TG :
                TG_list = TG.split(";")
                for single_TG in TG_list :
                    TFLink_Filtered_dict[TF].add(single_TG)
                    TFLink_Filtered_TFTGs.append((TF, single_TG))
                    TG_TF_graph.add_edge(single_TG, TF)
            else :
                TFLink_Filtered_dict[TF].add(TG)
                TFLink_Filtered_TFTGs.append((TF, TG))
                TG_TF_graph.add_edge(TG, TF)

        TFLink_Filtered_TFs = set(TFLink_Filtered_dict.keys())
        TFLink_Filtered_TGs = set(chain(*TFLink_Filtered_dict.values()))
        TFLink_Filtered_TFTGs = set(TFLink_Filtered_TFTGs)

        print(f"TFs : {len(TFLink_Filtered_TFs)}, TGs : {len(TFLink_Filtered_TGs)}")
    
        return TFLink_Filtered_TFTGs, TFLink_Filtered_TFs, TFLink_Filtered_TGs, TG_TF_graph
    
    print(f"External TF-TG database : TFLink (evidence cutoff : {evidence_cutoff})")
    
    #strong_TFTG_list, strong_TFs, strong_TGs, strong_TG_TF_graph = filter_TFLink(TFLink_All_df, True, evidence_cutoff=evidence_cutoff)
    #return strong_TFTG_list, strong_TFs, strong_TGs, strong_TG_TF_graph
    
    weak_TFTG_list, weak_TFs, weak_TGs, weak_TG_TF_graph = filter_TFLink(TFLink_All_df, False, evidence_cutoff=evidence_cutoff)
    return weak_TFTG_list, weak_TFs, weak_TGs, weak_TG_TF_graph

def read_pathway_reference() :
        
    ORGANISM = "hsa"
    KEGG_path = os.path.join(os.getcwd(), "data", "KEGG", ORGANISM)

    xml_ids = list(chain(*pd.read_csv(os.path.join(KEGG_path, "xml_ids.txt"), header=None).values.tolist()))
        
    KEGG_graph_nodes = {}
    KEGG_graph_edges = {}
    KEGG_graphs = {}

    KEGG_graph_nodes_noexp = {}
    KEGG_graph_edges_noexp = {}
    KEGG_graphs_noexp = {}
        
    KEGG_to_symbol_dict = {}
    symbol_to_pathway_dict = defaultdict(list)
    pathway_to_symbol_dict = defaultdict(frozenset)

    for pathway_id in xml_ids :

        if pathway_id.startswith(ORGANISM + "05") : continue # disease
        if pathway_id.startswith(ORGANISM + "015") : continue # drug resistance
        if pathway_id in ["hsa04930", "hsa04940", "hsa04950", "hsa04936", "hsa04932", "hsa04931", "hsa04933", "hsa04934"] : continue # endocrine and metabolic disease
        if pathway_id in ["hsa03250", "hsa03260", "hsa03264", "hsa03265", "hsa03266", "hsa03267"] : continue # viral
        if pathway_id in ["hsa04215", "hsa04213", "hsa04392"] : continue # multiple species
        if pathway_id in ["hsa04136"] : continue # other eukaryotes

        # for expansion
        
        node_file = ".".join(["path:" + pathway_id, "IdentifiersDictionary", "txt"])
        edge_file = ".".join(["path:" + pathway_id, "DirectedEdges", "GeneExpansion", "withTypes", "txt"])

        KEGG_graph_nodes[pathway_id] = pd.read_csv(os.path.join(KEGG_path, node_file), sep="\t")
        try: KEGG_graph_edges[pathway_id] = pd.read_csv(os.path.join(KEGG_path, edge_file), sep="\t")
        except EmptyDataError:
            KEGG_graph_edges[pathway_id] = pd.DataFrame(data={"edge" : [], "type" : [], "subtype" : []})
            continue

        KEGG_graph_nodes[pathway_id]["KEGG"] = ORGANISM + ":" + KEGG_graph_nodes[pathway_id]["KEGG"].astype(str)
        KEGG_to_symbol_dict.update(dict(zip(KEGG_graph_nodes[pathway_id]["KEGG"], KEGG_graph_nodes[pathway_id]["ENTREZ"])))

        for each_symbol in KEGG_graph_nodes[pathway_id]["ENTREZ"] : symbol_to_pathway_dict[each_symbol].append(pathway_id)
        pathway_to_symbol_dict[pathway_id] = set(KEGG_graph_nodes[pathway_id]["ENTREZ"])

        KEGG_graphs[pathway_id] = nx.DiGraph()

        for idx, row in KEGG_graph_edges[pathway_id].iterrows() :
            source, target = row["edge"].split("~")
            source, target = KEGG_to_symbol_dict[source], KEGG_to_symbol_dict[target]
            KEGG_graphs[pathway_id].add_edge(source, target, DB="KEGG", Level="Gene-Gene", Type=row["type"], Subtype=row["subtype"], Pathway=pathway_id)

        KEGG_graphs[pathway_id].remove_edges_from(list(nx.selfloop_edges(KEGG_graphs[pathway_id])))
        KEGG_graphs[pathway_id].remove_nodes_from(list(nx.isolates(KEGG_graphs[pathway_id])))        
            
        # for non-expansion
            
        node_file = ".".join(["path:" + pathway_id, "EntrytoGenes", "NoGeneExpansion", "txt"])
        edge_file = ".".join(["path:" + pathway_id, "DirectedEdges", "NoGeneExpansion", "withTypes", "txt"])

        KEGG_graph_nodes_noexp[pathway_id] = pd.read_csv(os.path.join(KEGG_path, node_file), sep="\t").applymap(str)
        try: KEGG_graph_edges_noexp[pathway_id] = pd.read_csv(os.path.join(KEGG_path, edge_file), sep="\t").applymap(str)
        except EmptyDataError: KEGG_graph_edges_noexp[pathway_id] = pd.DataFrame(data={"edge" : [], "type" : [], "subtype" : []})    

        entry_to_symbol_dict = dict(zip(KEGG_graph_nodes_noexp[pathway_id]["entry"], KEGG_graph_nodes_noexp[pathway_id]["group"]))

        KEGG_graphs_noexp[pathway_id] = nx.DiGraph()
        added_pairs = defaultdict(int)

        for idx, row in KEGG_graph_edges_noexp[pathway_id].iterrows() :

            source, target = row["edge"].split("~")

            if added_pairs[source, target] != 0 : continue
            added_pairs[source, target] += 1

            source_name = entry_to_symbol_dict[source]
            target_name = entry_to_symbol_dict[target]

            source_subnodes = source_name.split("*")
            source_subnodes = sorted([KEGG_to_symbol_dict[node] for node in source_subnodes])

            target_subnodes = target_name.split("*")
            target_subnodes = sorted([KEGG_to_symbol_dict[node] for node in target_subnodes])

            source_name = "*".join(source_subnodes)
            target_name = "*".join(target_subnodes)

            KEGG_graphs_noexp[pathway_id].add_edge(source_name, target_name, DB="KEGG", Type=row["type"], Subtype=row["subtype"], Pathway=pathway_id)
            KEGG_graphs_noexp[pathway_id].nodes[source_name]["Unit"] = source_subnodes
            KEGG_graphs_noexp[pathway_id].nodes[target_name]["Unit"] = target_subnodes

        KEGG_graphs_noexp[pathway_id].remove_edges_from(list(nx.selfloop_edges(KEGG_graphs_noexp[pathway_id])))
        KEGG_graphs_noexp[pathway_id].remove_nodes_from(list(nx.isolates(KEGG_graphs_noexp[pathway_id])))

    KEGG_G = nx.compose_all(KEGG_graphs.values())

    print(f"Pathways : {len(KEGG_graphs.keys())}")
    print(f"Genes : {KEGG_G.number_of_nodes()}, Edges : {KEGG_G.number_of_edges()}")
        
    return KEGG_graphs, KEGG_graphs_noexp, KEGG_G

def generate_TFTG_bridge(external_TFTGs=None) :
    
    effectors = defaultdict(set)
    TG_effectors = defaultdict(set)
    TG_parents = defaultdict(set)
    TFTG_relations = defaultdict(set)

    for pathway_id, pathway_graph in PATHWAY_GRAPHS_NOEXP.items() :

        if not pathway_id in KEGG_SIGNALING_PATHWAYS : continue

        leaf_nodes = set([node for node in pathway_graph.nodes if pathway_graph.out_degree(node) == 0])
        edges_with_types = nx.get_edge_attributes(pathway_graph, "Type")

        for effector_node in leaf_nodes :
            for parent_node in pathway_graph.predecessors(effector_node) :
                if edges_with_types[(parent_node, effector_node)] == "GErel" :
                    TG_effectors[pathway_id].add(effector_node)
                    TG_parents[pathway_id].add(parent_node)
                    TFTG_relations[pathway_id].add((parent_node, effector_node))
                    
        effectors[pathway_id] = set(filter(None, "*".join(list(leaf_nodes)).split("*"))) | set(filter(None, "*".join(list(TG_parents[pathway_id])).split("*")))
        TG_parents[pathway_id] = set(filter(None, "*".join(list(TG_parents[pathway_id])).split("*")))
        TG_effectors[pathway_id] = set(filter(None, "*".join(list(TG_effectors[pathway_id])).split("*")))
        
    pathway_TFTGs = set(chain(*TFTG_relations.values()))
    
    if not external_TFTGs is None : CASCADE_TFTGS = external_TFTGs
    else : CASCADE_TFTGS = pathway_TFTGs

    CASCADE_TFS = set([x[0] for x in CASCADE_TFTGS])
    CASCADE_TGS = set([x[1] for x in CASCADE_TFTGS])

    CASCADE_TFTG_DICT = defaultdict(list)
    for (TF, TG) in CASCADE_TFTGS : CASCADE_TFTG_DICT[TF].append(TG)

    print(f"CASCADE TF-TG relations : {len(CASCADE_TFTGS)}, CASCADE TFs : {len(CASCADE_TFS)}, CASCADE TGs : {len(CASCADE_TGS)}")
    
    return CASCADE_TFTG_DICT, CASCADE_TFS, effectors, TG_parents, TG_effectors


def generate_reachable_nodes() :
    
    reachable_dict = defaultdict(dict)
    reachable_effector = defaultdict(dict)
    reachable_target = defaultdict(dict)
    reachable_TF = defaultdict(dict)
    reachable_TG = defaultdict(dict)
    reachable_TFTG = defaultdict(dict)
    
    for pathway_id, pathway_graph in PATHWAY_GRAPHS_EXP.items() :

        for node in pathway_graph.nodes :
            
            reachable_nodes = set(nx.dfs_postorder_nodes(pathway_graph, source=node)) - set([node])
            reachable_effectors = PATHWAY_EFFECTORS[pathway_id] & reachable_nodes
            
            reachable_target[node][pathway_id] = sorted(list(reachable_nodes & TARGET_GENES))
            reachable_effector[node][pathway_id] = list(reachable_effectors)
            reachable_TG[node][pathway_id] = sorted(list(reachable_nodes & PATHWAY_TG_EFFECTORS[pathway_id]))
            reachable_TF[node][pathway_id] = sorted(list((reachable_nodes & PATHWAY_TG_PARENTS[pathway_id]) | (reachable_effectors & CASCADE_TF_GENES)))            
            reachable_TFTG[node][pathway_id] = list()
            for TF in reachable_TF[node][pathway_id] :
                for TG in sorted(CASCADE_TFTG_DICT[TF]) :
                    reachable_TFTG[node][pathway_id].append((TF, TG))

            reachable_dict[node][pathway_id] = dict()
            reachable_dict[node][pathway_id]["target"] = reachable_target[node][pathway_id]
            reachable_dict[node][pathway_id]["effector"] = reachable_effector[node][pathway_id]
            reachable_dict[node][pathway_id]["TFTG"] = reachable_TFTG[node][pathway_id]
            
    return reachable_dict, reachable_target, reachable_TF, reachable_TG, reachable_TFTG
                        
def subpathway_cascade_algorithm_v2(target_assay_gene, visited_nodes, visited_pathways, reachable_dicts, last_cascade="None") :
        
    if len(visited_pathways) - visited_pathways.count("TFTG") >= BLOCK_CUTOFF : 
        return ["Unreachable", "Unreachable"]
    
    for pathway_id in PATHWAY_IDS :
        
        if pathway_id in visited_pathways : continue
        if not pathway_id in reachable_dicts.keys() : continue
        
        # case 1 : directly reach to target assay gene
        if target_assay_gene in reachable_dicts[pathway_id]["target"] :
            yield visited_nodes + [target_assay_gene], visited_pathways + [pathway_id]
        
        # case 2-1 : natural cascade
        for effector in reachable_dicts[pathway_id]["effector"] :
            if effector in visited_nodes or effector == target_assay_gene : continue
            yield from subpathway_cascade_algorithm_v2(target_assay_gene=target_assay_gene,
                                                       visited_nodes=visited_nodes + [effector], 
                                                       visited_pathways=visited_pathways + [pathway_id], 
                                                       reachable_dicts=REACHABLE_DICT[effector],
                                                       last_cascade="natural")
        
        # case 2-2 : TF-TG cascade
        if visited_pathways.count("TFTG") < BRIDGE_CUTOFF :
            for (cascading_TF, cascading_TG) in reachable_dicts[pathway_id]["TFTG"] :
                if cascading_TF in visited_nodes or cascading_TG in visited_nodes or cascading_TF == target_assay_gene : continue
                if cascading_TG == target_assay_gene :
                    yield visited_nodes + [cascading_TF, cascading_TG], visited_pathways + [pathway_id, "TFTG"]
                else :
                    yield from subpathway_cascade_algorithm_v2(target_assay_gene=target_assay_gene,
                                                               visited_nodes=visited_nodes + [cascading_TF, cascading_TG], 
                                                               visited_pathways=visited_pathways + [pathway_id, "TFTG"], 
                                                               reachable_dicts=REACHABLE_DICT[cascading_TG],
                                                               last_cascade="TFTG")

                    
def generate_subpathway_blocks() :
    
    connected_targets = []
    subpathway_blocks = defaultdict(list)
    subpathway_pathways = defaultdict(list)
    
    print("Subpathway Cascade Started")
    with open(os.path.join(SUBPATHWAY_FOLDER_PATH, SUBPATHWAY_BLOCK_FILE_NAME), 'w') as f : f.write("\t".join(["Blocks", "Pathways"]) + "\n")
    with open(os.path.join(SUBPATHWAY_FOLDER_PATH, SUBPATHWAY_BLOCK_FILE_NAME), 'a') as f:
    
        for source, target in tqdm(list(itertools.product(SOURCE_GENES, TARGET_GENES)), desc=f"{len(connected_targets)}") :
            
            for bone_gene, bone_pathway in subpathway_cascade_algorithm_v2(target_assay_gene=target, 
                                                                           visited_nodes=[source], 
                                                                           visited_pathways=[], 
                                                                           reachable_dicts=REACHABLE_DICT[source]) :
                if bone_gene[-1] != target : continue
                
                f.write("*".join(bone_gene) + "\t"  + "-".join(bone_pathway) + "\n")
                if not target in connected_targets : connected_targets.append(target)
                subpathway_blocks[(source, target)].append(bone_gene)
                subpathway_pathways[(source, target)].append(bone_pathway)

    connected_targets = set(connected_targets)

    print()
    print()
    
    print(f"The number of target genes : {len(TARGET_GENES)}")
    print(f"The number of connected target genes by subpathway cascde : {len(connected_targets)}")
     
    print()
    print()
        
    with open(os.path.join(SUBPATHWAY_FOLDER_PATH, "DisconnectedTargetGenes.txt"), 'w') as f :
        print(f"Disconnected Genes : {len(TARGET_GENES - connected_targets)}")
        print("\n".join(sorted(list(TARGET_GENES - connected_targets))))
        f.write("\n".join(sorted(list(TARGET_GENES - connected_targets))))
        
    subpathway_blocks, subpathway_pathways, cascading_TFTGs = extract_candidate_subpathways(os.path.join(SUBPATHWAY_FOLDER_PATH, SUBPATHWAY_BLOCK_FILE_NAME))        
    block_components = extract_block_components(subpathway_blocks, subpathway_pathways)

    print()
    print()
    print(f"The number of subpathway blocks : {len(block_components)}")
    print(f"The number of TF-TG regulations : {cascading_TFTGs.number_of_nodes()}")

def extract_candidate_subpathways(subpathway_file) :
    
    subpathway_block_df = pd.read_csv(subpathway_file, sep="\t")

    subpathway_blocks = defaultdict(list)
    subpathway_pathways = defaultdict(list)

    for block_list, pathway_list in tqdm(zip(subpathway_block_df["Blocks"], subpathway_block_df["Pathways"])) :

        block_list = block_list.split("*")
        pathway_list = pathway_list.split("-")

        source = block_list[0]
        target = block_list[-1]

        subpathway_blocks[(source, target)].append(block_list)
        subpathway_pathways[(source, target)].append(pathway_list)

    TFTG_G = nx.DiGraph()

    for (source, target) in tqdm(subpathway_blocks.keys()) :
        for (p, pathways) in zip(subpathway_blocks[(source, target)], subpathway_pathways[(source, target)]) :
            for i, (gene1, gene2) in enumerate(zip(p[:-1], p[1:])) :
                if pathways[i] == "TFTG" :
                    TFTG_G.add_edge(gene1, gene2, Level="Gene-Gene", Type="TFTG", Subtype="None", Pathway="None")

    return subpathway_blocks, subpathway_pathways, TFTG_G
    
def extract_block_components(subpathway_blocks, subpathway_pathways) :
    
    block_components = list()
    checked_triplets = list()

    for (source, target) in tqdm(subpathway_blocks.keys()) :
        for (p, pathways) in zip(subpathway_blocks[(source, target)], subpathway_pathways[(source, target)]) :
            for i, (gene1, gene2) in enumerate(zip(p[:-1], p[1:])) :
                if not pathways[i] in ["TFTG", "TGTG"] :
                    if not [pathways[i], gene1, gene2] in checked_triplets :
                        checked_triplets.append([pathways[i], gene1, gene2])
                        block_components.append("_".join([pathways[i], gene1, gene2]))
    
    return block_components


if __name__ == "__main__" : 
    
    args = argument_parsing()

    HYPERPARAMETERS = [f"SOURCE={args.SOURCE_GENE_SET}", 
                       f"TARGET={args.TARGET_GENE_SET}", 
                       f"BLOCK={args.BLOCK_CUTOFF}", 
                       f"BRIDGE={args.BRIDGE_CUTOFF}",
                       f"EVIDENCE={args.EVIDENCE_CUTOFF}"]

    print()
    print("\n".join(HYPERPARAMETERS))

    SOURCE_GENE_SET = args.SOURCE_GENE_SET
    TARGET_GENE_SET = args.TARGET_GENE_SET
    BLOCK_CUTOFF = args.BLOCK_CUTOFF
    BRIDGE_CUTOFF = args.BRIDGE_CUTOFF
    EVIDENCE_CUTOFF = args.EVIDENCE_CUTOFF

    SUBPATHWAY_FOLDER_PATH = os.path.join(os.getcwd(), "SubpathwayCascade", ".".join(HYPERPARAMETERS))
    if not os.path.exists(SUBPATHWAY_FOLDER_PATH) : os.makedirs(SUBPATHWAY_FOLDER_PATH)
    SUBPATHWAY_BLOCK_FILE_NAME = "CascadeBackbones.txt"

    print()
    print()

    SOURCE_GENES, TARGET_GENES = read_source_target_gene_sets()
    TFTG_LIST, TF_GENES, TG_GENES, TG_TF_GRAPH = read_external_TFTG_reference(evidence_cutoff=EVIDENCE_CUTOFF)
    
    CASCADE_TF_GENES = TF_GENES

    print()
    print()

    PATHWAY_GRAPHS_EXP, PATHWAY_GRAPHS_NOEXP, PATHWAY_COMPOSED_GRAPH = read_pathway_reference()
    PATHWAY_IDS = PATHWAY_GRAPHS_EXP.keys()
    print()
    print()


    CASCADE_TFTG_DICT, CASCADE_TFS, PATHWAY_EFFECTORS, PATHWAY_TG_EFFECTORS, PATHWAY_TG_PARENTS = generate_TFTG_bridge(external_TFTGs=TFTG_LIST)
    print()
    print()

    REACHABLE_DICT, _, __, ___, ____ = generate_reachable_nodes()

    with open(os.path.join(SUBPATHWAY_FOLDER_PATH, "ReachableDict.pkl"), "wb") as fw : pickle.dump(REACHABLE_DICT, fw)
    print()
    print()
    
    generate_subpathway_blocks()
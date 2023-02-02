import argparse

import os
import numpy as np
import pandas as pd
import networkx as nx
import itertools

from tqdm import tqdm
from collections import defaultdict, Counter
from pandas.errors import EmptyDataError
from itertools import chain

import pickle

import torch
from torch_geometric.utils.convert import from_networkx
    
def read_target_gene_set(TARGET_GENE_SET) :
    
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
    
    return TARGET_GENES


def argument_parsing() :

    parser = argparse.ArgumentParser(description='SubpathwayCascade')

    parser.add_argument('--SOURCE_GENE_SET', type=str, help='source gene set name', default='CGC_Hallmark_BreastCancer')
    parser.add_argument('--TARGET_GENE_SET', type=str, help='target gene set name', default='UpdatedOncotypeDXCancer')
    
    parser.add_argument('--BLOCK_CUTOFF', type=int, help='cascade block cutoff', default=2)
    parser.add_argument('--BRIDGE_CUTOFF', type=int, help='bridge cutoff', default=1)
    parser.add_argument('--EVIDENCE_CUTOFF', type=int, help='TFTG evidence cutoff', default=5)
    
    parser.add_argument('--LENGTH_CUTOFF', type=int, help='block length cutoff', default=2)
    
    args = parser.parse_args()
    
    return args


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
        
    supernodes_in_pathway = defaultdict(nx.Graph)
    for pathway_id, pathway_graph in KEGG_graphs.items() :
        supernodes_in_pathway[pathway_id].add_nodes_from(KEGG_graphs[pathway_id].nodes)

    for pathway_id, pathway_graph in KEGG_graphs_noexp.items() :
        for supernode in pathway_graph.nodes :
            supernodes_in_pathway[pathway_id].add_edges_from(itertools.combinations(supernode.split("*"), 2))
        
    return KEGG_graphs, KEGG_graphs_noexp, supernodes_in_pathway


def extract_candidate_subpathways(subpathway_file, target_genes) :
    
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
    
    subpathway_landscape_G = nx.DiGraph()
    pathway_landscape_G = nx.DiGraph()
    assay_regulator_subpathway_G = nx.DiGraph()
    
    connected_assay_genes = set()

    for (source, target) in tqdm(subpathway_blocks.keys()) :
        for (p, pathways) in zip(subpathway_blocks[(source, target)], subpathway_pathways[(source, target)]) :
            subpathway_block_sequence = []
            pathway_sequence = []
            for i, (gene1, gene2) in enumerate(zip(p[:-1], p[1:])) :
                if pathways[i] == "TFTG" :
                    TFTG_G.add_edge(gene1, gene2, Level="Gene-Gene", Type="TFTG", Subtype="None", Pathway="None")
                else :
                    subpathway_block_sequence.append("_".join([pathways[i], gene1, gene2]))
                    pathway_sequence.append(pathways[i])
            if p[-1]  in target_genes :
                assay_regulator_subpathway_G.add_edge(subpathway_block_sequence[-1], "ASSAY_" + p[-1])
                connected_assay_genes.add("ASSAY_" + p[-1])
            nx.add_path(subpathway_landscape_G, subpathway_block_sequence)
            nx.add_path(pathway_landscape_G, pathway_sequence)
            
    return subpathway_blocks, subpathway_pathways, TFTG_G, subpathway_landscape_G, pathway_landscape_G, assay_regulator_subpathway_G, connected_assay_genes



def extract_block_graphs(subpathway_block_components, KEGG_graphs, graph_path) :
        
    pathways = []
    path_lengths = []
    gene_sets = []
    
    subgraphs = defaultdict(nx.DiGraph)
    allsp_graphs = defaultdict(nx.DiGraph)
    
    for block in tqdm(subpathway_block_components) : 
        
        pathway, source, target = block.split("_")
        
        pathways.append(pathway)
        path_lengths.append(nx.shortest_path_length(KEGG_graphs[pathway], source=source, target=target))
        
        allsp = nx.all_shortest_paths(G=KEGG_graphs[pathway], source=source, target=target)        
            
        gene_set = sorted(list(set(chain(*list(allsp)))))
        for sp in allsp : nx.add_path(allsp_graphs[block], sp)
        subgraphs[block] = KEGG_graphs[pathway].subgraph(gene_set).copy()
        
        gene_sets.append("*".join(gene_set))
                         
    block_df = pd.DataFrame(data={"BlockName" : subpathway_block_components,
                                  "Pathway" : pathways,
                                  "Length" : path_lengths,
                                  "Genes" : gene_sets}).set_index("BlockName")
    block_df.to_csv(os.path.join(graph_path, "SubpathwayBlocks.txt"), sep="\t")
    
    with open(os.path.join(graph_path, 'SubpathwayBlocks.UnionSP.pkl'), 'wb') as fw : pickle.dump(allsp_graphs, fw)
    with open(os.path.join(graph_path, 'SubpathwayBlocks.Subgraph.pkl'), 'wb') as fw : pickle.dump(subgraphs, fw)
        
    return block_df


def condense_isomorphic_blocks(supernodes_in_pathway, subpathway_block_names) :
    
    if type(subpathway_block_names) == list :
        subpathway_block_names = pd.Series(subpathway_block_names)
        
    subpathway_block_names.sort_values(inplace=True)
    isomorphic_subpathway_blocks_G = nx.Graph()
    isomorphic_subpathway_blocks_G.add_nodes_from(subpathway_block_names)
    
    for block1, block2 in tqdm(list(itertools.combinations(subpathway_block_names, 2))) :
        
        pathway1, source1, target1 = block1.split("_")
        pathway2, source2, target2 = block2.split("_")
        
        if pathway1 != pathway2 : continue
        
        if nx.has_path(supernodes_in_pathway[pathway1], source=source1, target=source2) and nx.has_path(supernodes_in_pathway[pathway1], source=target1, target=target2) : 
            isomorphic_subpathway_blocks_G.add_edge(block1, block2)

    condensed_subpathway_blocks = list()
    
    for component in nx.connected_components(isomorphic_subpathway_blocks_G) :
            
        component = sorted(list(component))
        pathway = component[0].split("_")[0]
        source_nodes = [block.split("_")[1] for block in component]
        target_nodes = [block.split("_")[2] for block in component]
        
        condensed_subpathway_blocks.append(["_".join([pathway, "+".join(source_nodes), "+".join(target_nodes)]), pathway, ";".join(source_nodes), ";".join(target_nodes), ";".join(component)])
            
    return pd.DataFrame(data=condensed_subpathway_blocks, columns=["BlockName", "Pathway", "Sources", "Targets", "SubpathwayBlocks"])


def filter_subpathways(subpathway_file, target_genes, mapped_graphs) :
    
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
    
    subpathway_landscape_G = nx.DiGraph()
    pathway_landscape_G = nx.DiGraph()
    assay_regulator_subpathway_G = nx.DiGraph()
    
    connected_assay_genes = set()

    for (source, target) in tqdm(subpathway_blocks.keys()) :
        for (p, pathways) in zip(subpathway_blocks[(source, target)], subpathway_pathways[(source, target)]) :
                        
            check_valid = True
            for i, (gene1, gene2) in enumerate(zip(p[:-1], p[1:])) :
                if pathways[i] != "TFTG" :
                    if not gene1 in mapped_graphs[pathways[i]].nodes : 
                        check_valid = False
                        break
                    if not gene2 in mapped_graphs[pathways[i]].nodes : 
                        check_valid = False
                        break
                    if not nx.has_path(mapped_graphs[pathways[i]], source=gene1, target=gene2) :
                        check_valid = False
                        break
                    if nx.shortest_path_length(mapped_graphs[pathways[i]], source=gene1, target=gene2) <= LENGTH_RESTRICTION :
                        check_valid = False
                        break
            if not check_valid : continue
            
            subpathway_block_sequence = []
            pathway_sequence = []
            
            for i, (gene1, gene2) in enumerate(zip(p[:-1], p[1:])) :
                if pathways[i] == "TFTG" :
                    TFTG_G.add_edge(gene1, gene2, Level="Gene-Gene", Type="TFTG", Subtype="None", Pathway="None")
                else :
                    subpathway_block_sequence.append("_".join([pathways[i], gene1, gene2]))
                    pathway_sequence.append(pathways[i])
                    
            if p[-1] in target_genes :
                assay_regulator_subpathway_G.add_edge(subpathway_block_sequence[-1], "ASSAY_" + p[-1])
                connected_assay_genes.add("ASSAY_" + p[-1])
                
            nx.add_path(subpathway_landscape_G, subpathway_block_sequence)
            nx.add_path(pathway_landscape_G, pathway_sequence)
            
    return subpathway_blocks, subpathway_pathways, TFTG_G, subpathway_landscape_G, pathway_landscape_G, assay_regulator_subpathway_G, connected_assay_genes


def generate_knowledge_graph_step1(graph_path, TFTG_G, subgraph=True) :
    
    if subgraph :
        with open(os.path.join(graph_path, 'SubpathwayBlocks.Subgraph.pkl'), 'rb') as f :
            gene_level_graphs = pickle.load(f)
    else :
        with open(os.path.join(graph_path, 'SubpathwayBlocks.UnionSP.pkl'), 'rb') as f :
            gene_level_graphs = pickle.load(f)
    
    gene_level_G = nx.compose_all(gene_level_graphs.values())
    gene_level_G = nx.compose_all([gene_level_G, TFTG_G])
    gene_to_subpathway_level_G = nx.DiGraph()
    
    for subpathway_block_name, subpathway_block_G in gene_level_graphs.items() :
        gene_to_subpathway_level_G.add_edges_from(list(zip(list(subpathway_block_G.nodes), [subpathway_block_name for _ in range(subpathway_block_G.number_of_nodes())])))
    
    return gene_level_G, gene_to_subpathway_level_G

def generate_knowledge_graph_step2(subpathway_landscape_G, assay_regulator_subpathway_G) :
    
    subpathway_level_G = subpathway_landscape_G
    subpathway_to_assay_level_G = assay_regulator_subpathway_G
    
    return subpathway_level_G, subpathway_to_assay_level_G


def condense_knowledge_graph(KG_dict, superblock_names, superblocks) :
    
    block_to_superblock = dict()
    
    for superblock_name, superblock in zip(superblock_names, superblocks) :
        for block in superblock :
            block_to_superblock[block] = superblock_name
    
    condensed_gene_to_subpathway_level_G = nx.DiGraph()
    condensed_subpathway_level_G = nx.DiGraph()
    condensed_subpathway_to_assay_level_G = nx.DiGraph()
    
    blocks = set(chain(*superblocks))
    
    # to include isolated nodes
    for node in KG_dict["Level-22"].nodes() :
        condensed_subpathway_level_G.add_node(block_to_superblock[node])

    for source, target in KG_dict["Level-12"].edges() :
        target = block_to_superblock[target]
        condensed_gene_to_subpathway_level_G.add_edge(source, target)
    for source, target in KG_dict["Level-22"].edges() :
        source = block_to_superblock[source]
        target = block_to_superblock[target]
        condensed_subpathway_level_G.add_edge(source, target)
    for source, target in KG_dict["Level-23"].edges() :
        source = block_to_superblock[source]
        condensed_subpathway_to_assay_level_G.add_edge(source, target)
        
    condensed_KG_dict = dict()
    condensed_KG_dict["Level-11"] = KG_dict["Level-11"]
    condensed_KG_dict["Level-12"] = condensed_gene_to_subpathway_level_G
    condensed_KG_dict["Level-22"] = condensed_subpathway_level_G
    condensed_KG_dict["Level-23"] = condensed_subpathway_to_assay_level_G
    
    condensed_KG_dict["Level-1.Nodes"] = set(condensed_KG_dict["Level-11"].nodes)
    condensed_KG_dict["Level-2.Nodes"] = set(condensed_KG_dict["Level-22"].nodes)
    condensed_KG_dict["Level-3.Nodes"] = KG_dict["Level-3.Nodes"]
    
    return condensed_KG_dict

def print_knowledge_graph_statistics(KG_dict) :
    
    print(f'Level-11, Nodes : {KG_dict["Level-11"].number_of_nodes()}, Edges : {KG_dict["Level-11"].number_of_edges()}')
    print(f'Level-12, Nodes : {KG_dict["Level-12"].number_of_nodes()}, Edges : {KG_dict["Level-12"].number_of_edges()}')
    print(f'Level-22, Nodes : {KG_dict["Level-22"].number_of_nodes()}, Edges : {KG_dict["Level-22"].number_of_edges()}')
    print(f'Level-23, Nodes : {KG_dict["Level-23"].number_of_nodes()}, Edges : {KG_dict["Level-23"].number_of_edges()}')
    print(f'Level-3, Nodes : {len(KG_dict["Level-3.Nodes"])}')    

if __name__ == "__main__" : 
    
    args = argument_parsing()

    HYPERPARAMETERS = [f"SOURCE={args.SOURCE_GENE_SET}", 
                       f"TARGET={args.TARGET_GENE_SET}", 
                       f"BLOCK={args.BLOCK_CUTOFF}", 
                       f"BRIDGE={args.BRIDGE_CUTOFF}",
                       f"EVIDENCE={args.EVIDENCE_CUTOFF}"]

    print()
    print("\n".join(HYPERPARAMETERS))
    print()
    print()

    SOURCE_GENE_SET = args.SOURCE_GENE_SET
    TARGET_GENE_SET = args.TARGET_GENE_SET
    BLOCK_CUTOFF = args.BLOCK_CUTOFF
    BRIDGE_CUTOFF = args.BRIDGE_CUTOFF
    EVIDENCE_CUTOFF = args.EVIDENCE_CUTOFF

    HYPERPARAMETERS = ".".join([f"SOURCE={SOURCE_GENE_SET}",
                                f"TARGET={TARGET_GENE_SET}",
                                f"BLOCK={BLOCK_CUTOFF}",
                                f"BRIDGE={BRIDGE_CUTOFF}",
                                f"EVIDENCE={EVIDENCE_CUTOFF}"])
    
    SUBPATHWAY_FOLDER_PATH = os.path.join(os.getcwd(), f"SubpathwayCascade", HYPERPARAMETERS)
    SUBPATHWAY_BLOCK_FILE_NAME = "CascadeBackbones.txt"

    SUBPATHWAY_GRAPH_PATH = os.path.join(SUBPATHWAY_FOLDER_PATH, "SubpathwayBlocks")
    if not os.path.exists(SUBPATHWAY_GRAPH_PATH) : os.makedirs(SUBPATHWAY_GRAPH_PATH, exist_ok=True)    

    TARGET_GENES = read_target_gene_set(TARGET_GENE_SET)    
    KEGG_graphs, KEGG_graphs_noexp, supernodes_in_pathway = read_pathway_reference()
    
    SUBPATHWAY_BLOCKS, SUBPATHWAY_PATHWAYS, TFTG_CASADE_G, SUBPATHWAY_LANDSCAPE_G, PATHWAY_LANDSCAPE_G, ASSAY_REGULATOR_SUBPATHWAY_G, CONNECTED_ASSAY_GENES = extract_candidate_subpathways(os.path.join(SUBPATHWAY_FOLDER_PATH, SUBPATHWAY_BLOCK_FILE_NAME), TARGET_GENES)
    BLOCK_NAMES = sorted(list(SUBPATHWAY_LANDSCAPE_G.nodes))
    CONDENSED_BLOCK_COMPONENTS_DF = condense_isomorphic_blocks(supernodes_in_pathway, BLOCK_NAMES)
        
    pd.DataFrame(PATHWAY_LANDSCAPE_G.edges(), columns=["source", "target"]).to_csv(os.path.join(SUBPATHWAY_GRAPH_PATH, "Landscape.Pathway.txt"), sep="\t", index=False)
    pd.DataFrame(SUBPATHWAY_LANDSCAPE_G.edges(), columns=["source", "target"]).to_csv(os.path.join(SUBPATHWAY_GRAPH_PATH, "Landscape.Subpathway.txt"), sep="\t", index=False)

    with open(os.path.join(SUBPATHWAY_GRAPH_PATH, "SubpathwayBlocksNames.txt"), "w") as f : f.write("\n".join(BLOCK_NAMES) + "\n")

    BLOCK_COMPONENTS_DF = extract_block_graphs(BLOCK_NAMES, KEGG_graphs, graph_path=SUBPATHWAY_GRAPH_PATH)
    REGULATORY_GENES = sorted(list(set(chain(*BLOCK_COMPONENTS_DF["Genes"].str.split("*"))) | TFTG_CASADE_G.nodes))
        
    with open(os.path.join(SUBPATHWAY_FOLDER_PATH, "RegulatoryGenes.txt"), 'w') as f : f.write("\n".join(sorted(REGULATORY_GENES)))
    CONDENSED_BLOCK_COMPONENTS_DF.to_csv(os.path.join(SUBPATHWAY_GRAPH_PATH, "SubpathwayBlocks.Condensed.txt"))
                
    KG_GENE, KG_GENE_TO_SUBPATHWAY = generate_knowledge_graph_step1(SUBPATHWAY_GRAPH_PATH, TFTG_CASADE_G)
    KG_SUBPATHWAY, KG_SUBPATHWAY_TO_ASSAY = generate_knowledge_graph_step2(SUBPATHWAY_LANDSCAPE_G, ASSAY_REGULATOR_SUBPATHWAY_G)
        
    KG_DICT = dict()
    KG_DICT["Level-11"] = KG_GENE
    KG_DICT["Level-12"] = KG_GENE_TO_SUBPATHWAY
    KG_DICT["Level-22"] = KG_SUBPATHWAY
    KG_DICT["Level-23"] = KG_SUBPATHWAY_TO_ASSAY
    KG_DICT["Level-1.Nodes"] = set(KG_DICT["Level-11"].nodes)
    KG_DICT["Level-2.Nodes"] = set(KG_DICT["Level-22"].nodes)
    KG_DICT["Level-3.Nodes"] = CONNECTED_ASSAY_GENES
        
    CONDENSED_KG_DICT = condense_knowledge_graph(KG_DICT, 
                                                 CONDENSED_BLOCK_COMPONENTS_DF["BlockName"], 
                                                 CONDENSED_BLOCK_COMPONENTS_DF["SubpathwayBlocks"].str.split(";"))
        
    with open(os.path.join(SUBPATHWAY_FOLDER_PATH, 'HierarchicalKnowledgeGraph.pkl'), 'wb') as fw : pickle.dump(KG_DICT, fw)
    with open(os.path.join(SUBPATHWAY_FOLDER_PATH, 'HierarchicalKnowledgeGraph.Condensed.pkl'), 'wb') as fw : pickle.dump(CONDENSED_KG_DICT, fw)
        
    print()
    print()
    
    print("Knowledge Graph (before condensation) :")
    print_knowledge_graph_statistics(KG_DICT)
        
    print()
    print()
        
    print("Knowledge Graph (after condensation) :")
    print_knowledge_graph_statistics(CONDENSED_KG_DICT)

    
    DATASET = "SCAN-B"
    FILTER_LOW_EXPRESSED_GENES = True
    LENGTH_RESTRICTION = args.LENGTH_CUTOFF
    FOLDER_PATH = os.path.join(os.getcwd(), "data", "MendeleySCAN-B")

    EXP_FOLDER_PATH = os.path.join(FOLDER_PATH, "StringTie FPKM Gene Data LibProtocol adjusted")
    EXP_FILE_NAME = "SCANB.9206.genematrix_noNeg.Symbol.txt"

    print()
    print()
    print(f"Mapping for {DATASET} started")
    print(f"Low expressed genes are filtered out and length restriction cutoff is {LENGTH_RESTRICTION}")
    
    exp = pd.read_csv(os.path.join(EXP_FOLDER_PATH, EXP_FILE_NAME), sep="\t")
    exp.set_index("Unnamed: 0", inplace=True)

    INPUT_GENES = sorted(list(set(exp.index)))

    if FILTER_LOW_EXPRESSED_GENES :
        exp = exp.groupby("Unnamed: 0").sum()
        exp = exp.loc[(exp == 0).sum(axis=1) / exp.shape[1] < 0.2]
        INPUT_GENES = sorted(list(set(exp.index)))
    
    KEGG_graphs, _, supernodes_in_pathway = read_pathway_reference()
    MAPPED_PATH = os.path.join(SUBPATHWAY_FOLDER_PATH, f"MAPPING={DATASET}.LENGTH={LENGTH_RESTRICTION}")
        
    if not os.path.exists(MAPPED_PATH) : os.makedirs(MAPPED_PATH, exist_ok=True)
        
    MAPPED_KEGG_GRAPHS = dict()
    for pathway_id, pathway_graph in KEGG_graphs.items() :    
        MAPPED_KEGG_GRAPHS[pathway_id] = KEGG_graphs[pathway_id].subgraph(INPUT_GENES)

    SUBPATHWAY_BLOCKS, SUBPATHWAY_PATHWAYS, TFTG_CASADE_G, SUBPATHWAY_LANDSCAPE_G, PATHWAY_LANDSCAPE_G, ASSAY_REGULATOR_SUBPATHWAY_G, CONNECTED_ASSAY_GENES = filter_subpathways(os.path.join(SUBPATHWAY_FOLDER_PATH, SUBPATHWAY_BLOCK_FILE_NAME), TARGET_GENES, MAPPED_KEGG_GRAPHS)

    BLOCK_NAMES = sorted(list(SUBPATHWAY_LANDSCAPE_G.nodes))

    CONDENSED_BLOCK_COMPONENTS_DF = condense_isomorphic_blocks(supernodes_in_pathway, BLOCK_NAMES)
    BLOCK_COMPONENTS_DF = extract_block_graphs(BLOCK_NAMES, MAPPED_KEGG_GRAPHS, graph_path=MAPPED_PATH)
    REGULATORY_GENES = sorted(list(set(chain(*BLOCK_COMPONENTS_DF["Genes"].str.split("*"))) | TFTG_CASADE_G.nodes))

    KG_GENE, KG_GENE_TO_SUBPATHWAY = generate_knowledge_graph_step1(MAPPED_PATH, TFTG_CASADE_G)
    KG_SUBPATHWAY, KG_SUBPATHWAY_TO_ASSAY = generate_knowledge_graph_step2(SUBPATHWAY_LANDSCAPE_G, ASSAY_REGULATOR_SUBPATHWAY_G)

    KG_DICT = dict()
    KG_DICT["Level-11"] = KG_GENE
    KG_DICT["Level-12"] = KG_GENE_TO_SUBPATHWAY
    KG_DICT["Level-22"] = KG_SUBPATHWAY
    KG_DICT["Level-23"] = KG_SUBPATHWAY_TO_ASSAY

    KG_DICT["Level-1.Nodes"] = set(KG_DICT["Level-11"].nodes)
    KG_DICT["Level-2.Nodes"] = set(KG_DICT["Level-22"].nodes)
    KG_DICT["Level-3.Nodes"] = CONNECTED_ASSAY_GENES

    CONDENSED_KG_DICT = condense_knowledge_graph(KG_DICT, 
                                                 CONDENSED_BLOCK_COMPONENTS_DF["BlockName"], 
                                                 CONDENSED_BLOCK_COMPONENTS_DF["SubpathwayBlocks"].str.split(";"))

    with open(os.path.join(MAPPED_PATH, 'HierarchicalKnowledgeGraph.pkl'), 'wb') as fw : pickle.dump(KG_DICT, fw)
    with open(os.path.join(MAPPED_PATH, 'HierarchicalKnowledgeGraph.Condensed.pkl'), 'wb') as fw : pickle.dump(CONDENSED_KG_DICT, fw)

    print("Knowledge Graph (before condensation) :")
    print_knowledge_graph_statistics(KG_DICT)

    print()
    print()

    print("Knowledge Graph (after condensation) :")
    print_knowledge_graph_statistics(CONDENSED_KG_DICT)


    # edge information
    level_11_G_without_attr = nx.DiGraph()
    level_22_G_without_attr = nx.DiGraph()
    level_22_G_without_attr.add_nodes_from(list(CONDENSED_KG_DICT["Level-22"].nodes()))

    level_11_G_without_attr.add_edges_from(list(CONDENSED_KG_DICT["Level-11"].edges()))
    level_22_G_without_attr.add_edges_from(list(CONDENSED_KG_DICT["Level-22"].edges()))

    level_11_G_int = nx.convert_node_labels_to_integers(level_11_G_without_attr, label_attribute="Name")
    level_22_G_int = nx.convert_node_labels_to_integers(level_22_G_without_attr, label_attribute="Name")

    level_11_G_torch = from_networkx(level_11_G_int)
    level_22_G_torch = from_networkx(level_22_G_int)

    level_1_nodes = level_11_G_torch.Name
    level_2_nodes = level_22_G_torch.Name
    level_3_nodes = list(CONDENSED_KG_DICT["Level-3.Nodes"])

    level_1_node_number = len(level_1_nodes)
    level_2_node_number = len(level_2_nodes)
    level_3_node_number = len(level_3_nodes)

    level_1_node_labels = dict(zip([i for i in range(level_1_node_number)], level_1_nodes))
    level_2_node_labels = dict(zip([i for i in range(level_2_node_number)], level_2_nodes))
    level_3_node_labels = dict(zip([i for i in range(level_3_node_number)], level_3_nodes))

    level_1_node_dict = dict(zip(level_1_nodes, [i for i in range(level_1_node_number)]))
    level_2_node_dict = dict(zip(level_2_nodes, [i for i in range(level_2_node_number)]))
    level_3_node_dict = dict(zip(level_3_nodes, [i for i in range(level_3_node_number)]))

    level_12_G = CONDENSED_KG_DICT["Level-12"]
    level_21_indicator = torch.zeros(level_2_node_number, level_1_node_number)
    for source, target in level_12_G.edges() :
        source = level_1_node_dict[source]
        target = level_2_node_dict[target]
        level_21_indicator[target][source] = 1

    level_23_G = CONDENSED_KG_DICT["Level-23"]
    level_32_indicator = torch.zeros(level_3_node_number, level_2_node_number)
    for source, target in level_23_G.edges() :
        source = level_2_node_dict[source]
        target = level_3_node_dict[target]
        level_32_indicator[target][source] = 1

    level_13_indicator = list()
    for assay_gene in level_3_nodes :
        level_13_indicator.append(level_1_nodes.index(assay_gene.split("_")[1]))
    
    torch_KG_dict = dict()
    
    torch_KG_dict["level_1_nodes"] = level_1_nodes
    torch_KG_dict["level_2_nodes"] = level_2_nodes
    torch_KG_dict["level_3_nodes"] = level_3_nodes
    
    torch_KG_dict["level_1_node_number"] = level_1_node_number
    torch_KG_dict["level_2_node_number"] = level_2_node_number
    torch_KG_dict["level_3_node_number"] = level_3_node_number
    
    torch_KG_dict["level_11_G_torch"] = level_11_G_torch
    torch_KG_dict["level_22_G_torch"] = level_22_G_torch
    torch_KG_dict["level_21_indicator"] = level_21_indicator
    torch_KG_dict["level_32_indicator"] = level_32_indicator
    torch_KG_dict["level_13_indicator"] = level_13_indicator
    
    torch.save(torch_KG_dict, os.path.join(MAPPED_PATH, "HierarchicalKnowledgeGraph.Condensed.TorchObject.pt"))
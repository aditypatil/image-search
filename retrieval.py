import os
import faiss
import pickle
import numpy as np
import models.clip_search as clip_search
import models.face_detection as face_detection
import models.geo_metadata as geo_metadata
import models.datetime_metadata as dtime
from datetime import datetime

import spacy
import pprint

class Search:

    def __init__(self, load_indices = True, embed_dir = 'embed_store'):

        self.embed_dir = embed_dir

        # loading up the indices from embed_dir
        self.img_path_index = pickle.load(open(os.path.join(self.embed_dir, 'img_path_index.pkl'), 'rb'))
        self.face_data = pickle.load(open(os.path.join(self.embed_dir, 'face_data.pkl'), 'rb'))
        self.flatten_img_face_index = pickle.load(open(os.path.join(self.embed_dir, 'img_path_index_for_face.pkl'), 'rb'))
        self.clip_embed = faiss.read_index(os.path.join(self.embed_dir, 'img_embeddings.bin'))
        self.geo_data = np.load(os.path.join(self.embed_dir, 'geo_metadata.npy'), allow_pickle=True)
        self.datetime_data = [datetime.fromisoformat(str(date)) for date in np.load(os.path.join(self.embed_dir, 'datetime_metadata.npy'), allow_pickle=True) if date is not None]
        

    def strategy1(self, query):

        # geo indices
        searchgeo = geo_metadata.SearchBM25()
        G_indices = searchgeo.search(query, geo_metadata=self.geo_data)

        # face indices to get F_indices
        searchface = face_detection.FaceSearchBM25(face_store=[self.face_data, self.flatten_img_face_index])
        F_indices = searchface.search(query= query)

        # strict sequential search strategy
        F_set = set(F_indices)
        G_set = set(G_indices)

        combined_indices = None

        # Apply the conditional logic
        if F_set and G_set:  # True if both sets are non-empty
            # Condition 1 met (both have results): Take intersection
            print('taking intersection...')
            combined_indices = F_set.intersection(G_set)
        elif F_set or G_set:
            print("taking union...")
            # Condition 2 met (at least one is empty, or both are empty): Take union
            combined_indices = F_set.union(G_set)
        else: 
            print("No location or people found. Doing semantic search via CLIP...")

        # clip search on combined_indices. If combined indices blank, then search through entire index. Else, search into index only on images searched by combined_indices
        if combined_indices:
            searchclip = clip_search.CLIPSearch(clip_embeddings=self.clip_embed, subset_id=list(combined_indices))
            C_indices = searchclip.search_faiss(query=query, top_k=10)
            return C_indices
        
        else:
            searchclip = clip_search.CLIPSearch(clip_embeddings=self.clip_embed)
            C_indices = searchclip.search_faiss(query=query, top_k=10)
            return C_indices
    
    def strategy2(self, query):

        
        # defining data struct for entities(person or location) in the query
        class EntityOps:

            def __init__(self, name = None, indices = None):
                self.name = name
                self.indices = indices
            
            def __repr__(self):
                return f"name: {self.name}; indices: {self.indices}"
        
        def geoindex_entities(query):
            '''finds if any geo entities are present by seraching the geo metadata. Returns list of EntityOps objects'''
            searchgeo = geo_metadata.SearchBM25()
            tokens = query.split(" ")

            geodata_entities = {}
            for token in tokens:
                token_filtered_indices = searchgeo.search(token, geo_metadata=self.geo_data)
                if len(token_filtered_indices) > 0:
                    geo_entity = EntityOps(name= token, indices= token_filtered_indices)
                    # geodata_entities.append(geo_entity)
                    geodata_entities[token] = geo_entity
            
            return geodata_entities

        def facedet_entities(query):
            '''finds if any face entities are present by seraching the geo metadata. Returns list of EntityOps objects'''
            searchface = face_detection.FaceSearchBM25(face_store=[self.face_data, self.flatten_img_face_index])
            tokens = query.split(" ")

            face_entities = {}
            for token in tokens:
                token_filtered_indices = searchface.search(query= token)
                if len(token_filtered_indices) > 0:
                    face_entity = EntityOps(name= token, indices= token_filtered_indices)
                    # face_entities.append(face_entity)
                    face_entities[token] = face_entity
            
            return face_entities
        
        def nlp_en_core_web(query):
            '''Runs spacy's en_core_web_lg model on the query. Returns spacy doc object'''

            spacy_model = "en_core_web_lg"
            nlp = spacy.load(spacy_model)

            doc = nlp(query)

            return doc

        # fetches dependency graph using spacy's 'en_core_web_lg' model. Corrects entities recognized by looking into face and geo metadata
        def get_corrected_entity_tokens(query):
            
            entity_tokens = {}

            query_doc = nlp_en_core_web(query= query)
            geoents = geoindex_entities(query= query)
            faceents = facedet_entities(query= query)
            faceent_names = list(faceents.keys())
            geoents_names = list(geoents.keys())
            none_entity = EntityOps()

            for token in query_doc:
                if token.text in faceent_names:
                    entity_tokens[token] = ['PERSON', faceents[token.text]]
                elif token.text in geoents_names:
                    entity_tokens[token] = ['LOCATION', geoents[token.text]]
                else:
                    entity_tokens[token] = ['NA', none_entity]

            return entity_tokens

        # entity_token = get_corrected_entity_tokens(query= query)
        # pprint.pprint(entity_token)
        
        
        def extract_entity_paths(query):

            entity_tokens_map = get_corrected_entity_tokens(query)

            # Combine all entity tokens for easier iteration
            all_entity_tokens = list(entity_tokens_map.keys())

            # 2. Traverse up the heads for each entity and build paths
            paths_info = []

            for entity_token in all_entity_tokens:
                path = []
                current_token = entity_token

                # Traverse up the dependency tree
                while current_token:
                    path.append({
                        "token": current_token.text,
                        "pos": current_token.pos_,
                        "dep": current_token.dep_,
                        "head": current_token.head.text if current_token.head else "ROOT",
                        "indices": entity_tokens_map[current_token][1].indices,
                        # "children": [(child.text, child.dep_) for child in current_token.children],
                        "children": [child for child in current_token.children],
                        "entity_category": entity_tokens_map[current_token][0]
                    })

                    # Stop if we hit another entity token (but not the starting one)
                    if current_token != entity_token and current_token in entity_tokens_map:
                        break # Stop when another entity from our list is found

                    # Move to the head of the current token
                    if current_token.head == current_token: # Catch root loop
                        break
                    current_token = current_token.head

                paths_info.append({
                    "start_entity": entity_token.text,
                    "start_entity_indices": entity_tokens_map[entity_token][1].indices,
                    "entity_category": entity_tokens_map[entity_token][0],
                    "path_to_root_or_other_entity": path
                })


            return paths_info
        



        entity_paths_info = extract_entity_paths(query= query)

        faceents = facedet_entities(query= query)
        faceent_names = list(faceents.keys())

        or_detected = 0
        if len(faceent_names) > 1:

            n_n_trees = [ entity_paths
                        for entity_paths in entity_paths_info
                            if (
                                (entity_paths['entity_category'] == 'PERSON' and
                                entity_paths['path_to_root_or_other_entity'][-1]['entity_category'] == 'PERSON')
                                and
                                (entity_paths['start_entity'] != entity_paths['path_to_root_or_other_entity'][-1]['token'])
                                )
                        ]
            
            name_sets = []
            for tree in n_n_trees:
                # print(f"{tree['start_entity']} AND {tree['path_to_root_or_other_entity'][-1]['token']}")
                # print(tree)
                
                if tree['path_to_root_or_other_entity'][0]['head'] in faceent_names:
                    children = [ child.text for child in tree['path_to_root_or_other_entity'][-1]['children']]
                    if 'or' in children:
                        ent_set = set(tree['start_entity_indices']).union(set(tree['path_to_root_or_other_entity'][-1]['indices']))
                        # print(f"{tree['start_entity']} OR {tree['path_to_root_or_other_entity'][-1]['token']}")
                        or_detected += 1
                    else:
                        ent_set = set(tree['start_entity_indices']).intersection(set(tree['path_to_root_or_other_entity'][-1]['indices']))
                        # print(f"{tree['start_entity']} AND {tree['path_to_root_or_other_entity'][-1]['token']}")


                else:
                    ent_set = set(tree['start_entity_indices']).intersection(set(tree['path_to_root_or_other_entity'][-1]['indices']))
                    # print(f"{tree['start_entity']} AND {tree['path_to_root_or_other_entity'][-1]['token']}")
                
                name_sets.append(ent_set)
            
            if or_detected == 1:
                final_name_set = set.union(*name_sets)
    
            else:
                final_name_set = set.intersection(*name_sets)
        
        elif len(faceent_names) == 1:
            n_n_trees = [ entity_paths
                        for entity_paths in entity_paths_info
                            if (
                                entity_paths['entity_category'] == 'PERSON'
                                )
                        ]
            
            name_sets = []
            for tree in n_n_trees:
                # print(f"{tree['start_entity']} OR {tree['path_to_root_or_other_entity'][-1]['token']}")
                ent_set = set(tree['start_entity_indices'])
                name_sets.append(ent_set)
            
            if or_detected == 1:
                final_name_set = set.union(*name_sets)
    
            else:
                final_name_set = set.intersection(*name_sets)
        
        else:
            final_name_set = set()


        geoents = geoindex_entities(query= query)
        geoent_names = list(geoents.keys())

        if len(geoent_names) > 1:
            l_l_trees = [ entity_paths
                        for entity_paths in entity_paths_info
                            if (
                                (entity_paths['entity_category'] == 'LOCATION' and
                                entity_paths['path_to_root_or_other_entity'][-1]['entity_category'] == 'LOCATION')
                                and
                                (entity_paths['start_entity'] != entity_paths['path_to_root_or_other_entity'][-1]['token'])
                                )
                        ]
            
            loc_sets = []
            for tree in l_l_trees:
                # print(f"{tree['start_entity']} OR {tree['path_to_root_or_other_entity'][-1]['token']}")
                ent_set = set(tree['start_entity_indices']).union(set(tree['path_to_root_or_other_entity'][-1]['indices']))
                loc_sets.append(ent_set)

            final_loc_set = set.union(*loc_sets)    

            
        elif len(geoent_names) == 1:
            l_l_trees = [ entity_paths
                        for entity_paths in entity_paths_info
                            if (
                                (entity_paths['entity_category'] == 'LOCATION')
                                )
                        ]
            
            loc_sets = []
            for tree in l_l_trees:
                # print(f"{tree['start_entity']} OR {tree['path_to_root_or_other_entity'][-1]['token']}")
                ent_set = set(tree['start_entity_indices'])
                loc_sets.append(ent_set)
            
            final_loc_set = set.union(*loc_sets)
        
        else:
            final_loc_set = set()
        
        
        if final_loc_set and final_name_set:
            combined_indices = final_loc_set.intersection(final_name_set)
        elif final_name_set or final_loc_set:
            combined_indices = final_loc_set.union(final_name_set)
        else:
            combined_indices = set()

        # print(f"combined name and loc indices: {combined_indices}")
        if combined_indices:
            searchclip = clip_search.CLIPSearch(clip_embeddings=self.clip_embed, subset_id=list(combined_indices))
            C_indices = searchclip.search_faiss(query=query, top_k=10)
            return C_indices
        
        elif ((len(combined_indices) == 0) and (len(geoent_names) > 0) and (len(faceent_names) > 0) ):
            
            return list(combined_indices)

        else:
            searchclip = clip_search.CLIPSearch(clip_embeddings=self.clip_embed)
            C_indices = searchclip.search_faiss(query=query, top_k=10)
            return C_indices
    
    
    
    
    
    def strategy3(self, query):
        
        # defining data struct for entities(person or location) in the query
        class EntityOps:

            def __init__(self, name = None, indices = None):
                self.name = name
                self.indices = indices
            
            def __repr__(self):
                return f"name: {self.name}; indices: {self.indices}"
        
        def geoindex_entities(query):
            '''finds if any geo entities are present by seraching the geo metadata. Returns list of EntityOps objects'''
            searchgeo = geo_metadata.SearchBM25()
            tokens = query.split(" ")

            geodata_entities = {}
            for token in tokens:
                token_filtered_indices = searchgeo.search(token, geo_metadata=self.geo_data)
                if len(token_filtered_indices) > 0:
                    geo_entity = EntityOps(name= token, indices= token_filtered_indices)
                    # geodata_entities.append(geo_entity)
                    geodata_entities[token] = geo_entity
            
            return geodata_entities

        def facedet_entities(query):
            '''finds if any face entities are present by seraching the geo metadata. Returns list of EntityOps objects'''
            searchface = face_detection.FaceSearchBM25(face_store=[self.face_data, self.flatten_img_face_index])
            tokens = query.split(" ")

            face_entities = {}
            for token in tokens:
                token_filtered_indices = searchface.search(query= token)
                if len(token_filtered_indices) > 0:
                    face_entity = EntityOps(name= token, indices= token_filtered_indices)
                    # face_entities.append(face_entity)
                    face_entities[token] = face_entity
            
            return face_entities
        
        def nlp_en_core_web(query):
            '''Runs spacy's en_core_web_lg model on the query. Returns spacy doc object'''

            spacy_model = "en_core_web_lg"
            nlp = spacy.load(spacy_model)

            doc = nlp(query)

            return doc

        # fetches dependency graph using spacy's 'en_core_web_lg' model. Corrects entities recognized by looking into face and geo metadata
        def get_corrected_entity_tokens(query):
            
            entity_tokens = {}

            query_doc = nlp_en_core_web(query= query)
            geoents = geoindex_entities(query= query)
            faceents = facedet_entities(query= query)
            faceent_names = list(faceents.keys())
            geoents_names = list(geoents.keys())
            none_entity = EntityOps()

            for token in query_doc:
                if token.text in faceent_names:
                    entity_tokens[token] = ['PERSON', faceents[token.text]]
                elif token.text in geoents_names:
                    entity_tokens[token] = ['LOCATION', geoents[token.text]]
                else:
                    entity_tokens[token] = ['NA', none_entity]

            return entity_tokens

        # entity_token = get_corrected_entity_tokens(query= query)
        # pprint.pprint(entity_token)
        
        
        def extract_entity_paths(query):

            entity_tokens_map = get_corrected_entity_tokens(query)

            # Combine all entity tokens for easier iteration
            all_entity_tokens = list(entity_tokens_map.keys())

            # 2. Traverse up the heads for each entity and build paths
            paths_info = []

            for entity_token in all_entity_tokens:
                path = []
                current_token = entity_token

                # Traverse up the dependency tree
                while current_token:
                    path.append({
                        "token": current_token.text,
                        "pos": current_token.pos_,
                        "dep": current_token.dep_,
                        "head": current_token.head.text if current_token.head else "ROOT",
                        "indices": entity_tokens_map[current_token][1].indices,
                        # "children": [(child.text, child.dep_) for child in current_token.children],
                        "children": [child for child in current_token.children],
                        "entity_category": entity_tokens_map[current_token][0]
                    })

                    # Stop if we hit another entity token (but not the starting one)
                    if current_token != entity_token and current_token in entity_tokens_map:
                        break # Stop when another entity from our list is found

                    # Move to the head of the current token
                    if current_token.head == current_token: # Catch root loop
                        break
                    current_token = current_token.head

                paths_info.append({
                    "start_entity": entity_token.text,
                    "start_entity_indices": entity_tokens_map[entity_token][1].indices,
                    "entity_category": entity_tokens_map[entity_token][0],
                    "path_to_root_or_other_entity": path
                })


            return paths_info
        


        duckling = dtime.DucklingEngine(port=8010)
        dt_search = dtime.DateSearch(self.datetime_data)
        dtime_indices = dt_search.search(duckling.get_response(query))
        del duckling
        print("Duckling indices: {dtime_indices}")
        

        entity_paths_info = extract_entity_paths(query= query)

        faceents = facedet_entities(query= query)
        faceent_names = list(faceents.keys())

        or_detected = 0
        if len(faceent_names) > 1:

            n_n_trees = [ entity_paths
                        for entity_paths in entity_paths_info
                            if (
                                (entity_paths['entity_category'] == 'PERSON' and
                                entity_paths['path_to_root_or_other_entity'][-1]['entity_category'] == 'PERSON')
                                and
                                (entity_paths['start_entity'] != entity_paths['path_to_root_or_other_entity'][-1]['token'])
                                )
                        ]
            
            name_sets = []
            for tree in n_n_trees:
                # print(f"{tree['start_entity']} AND {tree['path_to_root_or_other_entity'][-1]['token']}")
                # print(tree)
                
                if tree['path_to_root_or_other_entity'][0]['head'] in faceent_names:
                    children = [ child.text for child in tree['path_to_root_or_other_entity'][-1]['children']]
                    if 'or' in children:
                        ent_set = set(tree['start_entity_indices']).union(set(tree['path_to_root_or_other_entity'][-1]['indices']))
                        # print(f"{tree['start_entity']} OR {tree['path_to_root_or_other_entity'][-1]['token']}")
                        or_detected += 1
                    else:
                        ent_set = set(tree['start_entity_indices']).intersection(set(tree['path_to_root_or_other_entity'][-1]['indices']))
                        # print(f"{tree['start_entity']} AND {tree['path_to_root_or_other_entity'][-1]['token']}")


                else:
                    ent_set = set(tree['start_entity_indices']).intersection(set(tree['path_to_root_or_other_entity'][-1]['indices']))
                    # print(f"{tree['start_entity']} AND {tree['path_to_root_or_other_entity'][-1]['token']}")
                
                name_sets.append(ent_set)
            
            if or_detected == 1:
                final_name_set = set.union(*name_sets)
    
            else:
                final_name_set = set.intersection(*name_sets)
        
        elif len(faceent_names) == 1:
            n_n_trees = [ entity_paths
                        for entity_paths in entity_paths_info
                            if (
                                entity_paths['entity_category'] == 'PERSON'
                                )
                        ]
            
            name_sets = []
            for tree in n_n_trees:
                # print(f"{tree['start_entity']} OR {tree['path_to_root_or_other_entity'][-1]['token']}")
                ent_set = set(tree['start_entity_indices'])
                name_sets.append(ent_set)
            
            if or_detected == 1:
                final_name_set = set.union(*name_sets)
    
            else:
                final_name_set = set.intersection(*name_sets)
        
        else:
            final_name_set = set()


        geoents = geoindex_entities(query= query)
        geoent_names = list(geoents.keys())

        if len(geoent_names) > 1:
            l_l_trees = [ entity_paths
                        for entity_paths in entity_paths_info
                            if (
                                (entity_paths['entity_category'] == 'LOCATION' and
                                entity_paths['path_to_root_or_other_entity'][-1]['entity_category'] == 'LOCATION')
                                and
                                (entity_paths['start_entity'] != entity_paths['path_to_root_or_other_entity'][-1]['token'])
                                )
                        ]
            
            loc_sets = []
            for tree in l_l_trees:
                # print(f"{tree['start_entity']} OR {tree['path_to_root_or_other_entity'][-1]['token']}")
                ent_set = set(tree['start_entity_indices']).union(set(tree['path_to_root_or_other_entity'][-1]['indices']))
                loc_sets.append(ent_set)

            final_loc_set = set.union(*loc_sets)    

            
        elif len(geoent_names) == 1:
            l_l_trees = [ entity_paths
                        for entity_paths in entity_paths_info
                            if (
                                (entity_paths['entity_category'] == 'LOCATION')
                                )
                        ]
            
            loc_sets = []
            for tree in l_l_trees:
                # print(f"{tree['start_entity']} OR {tree['path_to_root_or_other_entity'][-1]['token']}")
                ent_set = set(tree['start_entity_indices'])
                loc_sets.append(ent_set)
            
            final_loc_set = set.union(*loc_sets)
        
        else:
            final_loc_set = set()
        
        

        # HANDLING INDICES ACROSS SEARCH RESULTS
        dtime_indices = set(dtime_indices)

        # Combine all sets based on availability
        all_sets = [s for s in [final_loc_set, final_name_set, dtime_indices] if s]

        if len(all_sets) == 3:
            combined_indices = all_sets[0].intersection(*all_sets[1:])
        elif len(all_sets) > 0:
            combined_indices = set().union(*all_sets)
        else:
            combined_indices = set()

        # Debugging (optional)
        # print(f"combined indices (name, loc, tag): {combined_indices}")

        # Search logic
        if combined_indices:
            searchclip = clip_search.CLIPSearch(clip_embeddings=self.clip_embed, subset_id=list(combined_indices))
            C_indices = searchclip.search_faiss(query=query, top_k=10)
            return C_indices

        elif ((len(combined_indices) == 0) and (len(geoent_names) > 0) and (len(faceent_names) > 0)):
            return list(combined_indices)

        else:
            searchclip = clip_search.CLIPSearch(clip_embeddings=self.clip_embed)
            C_indices = searchclip.search_faiss(query=query, top_k=10)
            return C_indices

        



def __main__(query):

    srch = Search()
    img_idx = srch.strategy3(query)
    # print(img_idx)
    return img_idx



    # srch = Search()
    # img_indices = srch.strategy1(query)
    # return img_indices
    

if __name__ == '__main__':
    __main__()

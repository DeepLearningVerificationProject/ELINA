#include "backsubstitute.h"


void update_state_using_predecessor_layer(fppoly_internal_t *pr,fppoly_t *fp, expr_t **lexpr_ptr, expr_t **uexpr_ptr, size_t k){
	expr_t *lexpr = *lexpr_ptr;
	expr_t *uexpr = *uexpr_ptr;
	expr_t *tmp_l = lexpr;
	expr_t *tmp_u = uexpr;
        neuron_t ** aux_neurons = fp->layers[k]->neurons; 
        *lexpr_ptr = lexpr_replace_bounds(pr,lexpr,aux_neurons, fp->layers[k]->is_activation);            
	*uexpr_ptr = uexpr_replace_bounds(pr,uexpr,aux_neurons, fp->layers[k]->is_activation);
	free_expr(tmp_l);
	free_expr(tmp_u);
}


void * update_state_using_previous_layers(void *args){
	nn_thread_t * data = (nn_thread_t *)args;
	elina_manager_t * man = data->man;
	fppoly_t *fp = data->fp;
	fppoly_internal_t * pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	size_t layerno = data->layerno;
	size_t idx_start = data->start;
	size_t idx_end = data->end;
	size_t i;
	int k;
	
	neuron_t ** out_neurons = fp->layers[layerno]->neurons;
	size_t num_out_neurons = fp->layers[layerno]->dims;
	
	for(i=idx_start; i < idx_end; i++){
		bool already_computed= false;
		expr_t *lexpr = copy_expr(out_neurons[i]->lexpr);
		expr_t *uexpr = copy_expr(out_neurons[i]->uexpr);
	 	if(fp->layers[layerno]->num_predecessors==2){
	 		
			expr_t * lexpr_copy = copy_expr(lexpr);
			lexpr_copy->inf_cst = 0;
			lexpr_copy->sup_cst = 0;
			expr_t * uexpr_copy = copy_expr(uexpr);
			uexpr_copy->inf_cst = 0;
			uexpr_copy->sup_cst = 0;
			size_t predecessor1 = fp->layers[layerno]->predecessors[0]-1;
			size_t predecessor2 = fp->layers[layerno]->predecessors[1]-1;
			char * predecessor_map = (char *)calloc(layerno,sizeof(char));
			// Assume no nested residual layers
			int iter = predecessor1;
			while(iter>=0){
				predecessor_map[iter] = 1;
				iter = fp->layers[iter]->predecessors[0]-1;
			}
			iter =  predecessor2;
			int common_predecessor = 0;
			while(iter>=0){
				if(predecessor_map[iter] == 1){
					common_predecessor = iter;
					break;
				}
				iter = fp->layers[iter]->predecessors[0]-1;
			}
				
			iter = predecessor1;
			while(iter!=common_predecessor){
				update_state_using_predecessor_layer(pr,fp, &lexpr, &uexpr, iter);
					
				iter = fp->layers[iter]->predecessors[0]-1;
			}
			iter =  predecessor2;
			while(iter!=common_predecessor){
					
				update_state_using_predecessor_layer(pr,fp, &lexpr_copy, &uexpr_copy, iter);
					
				iter = fp->layers[iter]->predecessors[0]-1;					
			}
			free(predecessor_map);
			add_expr(pr,lexpr,lexpr_copy);
			add_expr(pr,uexpr,uexpr_copy);
			free_expr(lexpr_copy);
			free_expr(uexpr_copy);
				// Assume at least one non-residual layer between two residual layers
				
			k = common_predecessor;
			
			
		}
		else{
			
			k = fp->layers[layerno]->predecessors[0]-1;
		}
		//printf("k: %d layerno: %zu\n",k,layerno);
		//fflush(stdout);
		while(k>=0){
		//for(k=fp->layers[layerno]->predecessors[0]-1; k >=0; k = fp->layers[k]->predecessors[0]-1){	
			neuron_t ** aux_neurons = fp->layers[k]->neurons;
			
			if(fp->layers[k]->num_predecessors==2){
				expr_t * lexpr_copy = copy_expr(lexpr);
				lexpr_copy->inf_cst = 0;
				lexpr_copy->sup_cst = 0;
				expr_t * uexpr_copy = copy_expr(uexpr);
				uexpr_copy->inf_cst = 0;
				uexpr_copy->sup_cst = 0;
				size_t predecessor1 = fp->layers[k]->predecessors[0]-1;
				size_t predecessor2 = fp->layers[k]->predecessors[1]-1;
				
				char * predecessor_map = (char *)calloc(k,sizeof(char));
				// Assume no nested residual layers
				int iter = predecessor1;
				while(iter>=0){
					predecessor_map[iter] = 1;
					iter = fp->layers[iter]->predecessors[0]-1;
				}
				iter =  predecessor2;
				int common_predecessor = 0;
				while(iter>=0){
					if(predecessor_map[iter] == 1){
						common_predecessor = iter;
						break;
					}
					iter = fp->layers[iter]->predecessors[0]-1;
				}
				
				iter = predecessor1;
				while(iter!=common_predecessor){
					
					update_state_using_predecessor_layer(pr,fp, &lexpr, &uexpr, iter);
					
					iter = fp->layers[iter]->predecessors[0]-1;
				}
				iter =  predecessor2;
				while(iter!=common_predecessor){
					
					update_state_using_predecessor_layer(pr,fp, &lexpr_copy, &uexpr_copy, iter);
					
					iter = fp->layers[iter]->predecessors[0]-1;					
				}
				free(predecessor_map);
				add_expr(pr,lexpr,lexpr_copy);
				add_expr(pr,uexpr,uexpr_copy);
				free_expr(lexpr_copy);
				free_expr(uexpr_copy);
				// Assume at least one non-residual layer between two residual layers
				
				k = common_predecessor;
				out_neurons[i]->lb = compute_lb_from_expr(pr, lexpr,fp,k); 
				out_neurons[i]->ub = compute_ub_from_expr(pr, uexpr,fp,k);
				already_computed = true;
				break;
				//continue;
			}
			else {
				
				 update_state_using_predecessor_layer(pr,fp, &lexpr, &uexpr, k);
				 k = fp->layers[k]->predecessors[0]-1;
				 
			}
			
			//printf("k %d\n",k);
			
		}
		
		if(!already_computed){
			out_neurons[i]->lb = compute_lb_from_expr(pr, lexpr,fp,-1); 
			//- bias_i;
			out_neurons[i]->ub = compute_ub_from_expr(pr, uexpr,fp,-1); //+ bias_i;
		}
		if(fp->out!=NULL){
			
			fp->out->lexpr[i] = lexpr;
			fp->out->uexpr[i] = uexpr;
		}
		else{
			free_expr(lexpr);
			free_expr(uexpr);
		}
		
	}
	
	return NULL;
}


void update_state_using_previous_layers_parallel(elina_manager_t *man, fppoly_t *fp, size_t layerno){
	//size_t NUM_THREADS = get_nprocs();
  	size_t NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
	nn_thread_t args[NUM_THREADS];
	pthread_t threads[NUM_THREADS];
	size_t num_out_neurons = fp->layers[layerno]->dims;
	size_t i;
	//printf("layerno %zu %zu\n",layerno,fp->layers[layerno]->predecessors[0]-1);
	//fflush(stdout);
	if(num_out_neurons < NUM_THREADS){
		for (i = 0; i < num_out_neurons; i++){
	    		args[i].start = i; 
	    		args[i].end = i+1;   
			args[i].man = man;
			args[i].fp = fp;
			args[i].layerno = layerno;
	    		pthread_create(&threads[i], NULL,update_state_using_previous_layers, (void*)&args[i]);
			
	  	}
		for (i = 0; i < num_out_neurons; i = i + 1){
			pthread_join(threads[i], NULL);
		}
	}
	else{
		size_t idx_start = 0;
		size_t idx_n = num_out_neurons / NUM_THREADS;
		size_t idx_end = idx_start + idx_n;
		
		
	  	for (i = 0; i < NUM_THREADS; i++){
	    		args[i].start = idx_start; 
	    		args[i].end = idx_end;   
			args[i].man = man;
			args[i].fp = fp;
			args[i].layerno = layerno;
	    		pthread_create(&threads[i], NULL,update_state_using_previous_layers, (void*)&args[i]);
			idx_start = idx_end;
			idx_end = idx_start + idx_n;
	    		if(idx_end>num_out_neurons){
				idx_end = num_out_neurons;
			}
			if((i==NUM_THREADS-2)){
				idx_end = num_out_neurons;
				
			}
	  	}
		//idx_start = idx_end;
	    	//idx_end = num_out_neurons;
		//args[i].start = idx_start; 
	    	//args[i].end = idx_end;   
		//args[i].man = man;
		//args[i].fp = fp;
		//args[i].layerno = layerno;
	    	//pthread_create(&threads[i], NULL,update_state_using_previous_layers, (void*)&args[i]);
		for (i = 0; i < NUM_THREADS; i = i + 1){
			pthread_join(threads[i], NULL);
		}
	}
	//printf("end\n");
	//fflush(stdout);
}

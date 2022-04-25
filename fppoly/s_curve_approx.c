#include "s_curve_approx.h"

enum s_curve_type {
    sigmoid,
    tanh_curve,
    elu_curve,
};

void compute_chord_slope(double *slope_inf, double *slope_sup, double f_sup_l, double f_sup_u,
			 double f_inf_l, double f_inf_u, double inf_l, double inf_u, double sup_l, double sup_u){
	double num_l =  f_sup_l + f_inf_u;
	double num_u =  f_sup_u + f_inf_l;

	double den_l = sup_l + inf_u;
	double den_u = sup_u + inf_l;
    	elina_double_interval_div(slope_inf, slope_sup, num_l, num_u, den_l, den_u);
}

void compute_derivative(double *slope_inf, double *slope_sup, double s_curve_l, double s_curve_u, double sq_l, double sq_u, enum s_curve_type s_curve){
    double sq_den_sup_l, sq_den_sup_u;
    elina_double_interval_mul(&sq_den_sup_l, &sq_den_sup_u, sq_l, sq_u, sq_l, sq_u);
    if(s_curve == sigmoid){
            elina_double_interval_div(slope_inf, slope_sup, s_curve_l, s_curve_u, sq_den_sup_l, sq_den_sup_u);
    }
    else if(s_curve == tanh_curve){
            *slope_inf = -1 + sq_den_sup_u;
            *slope_sup = 1 + sq_den_sup_l;
    }
    else if(s_curve == elu_curve){
        // For ELU, s_curve_l = sq_l = f_{inf or sup}_l = elu(...), and s_curve_u = sq_u = f_{inf or sup}_u = elu(...)
        // TODO(klinvill): Is checking the s_curve values the proper way to ensure which region of ELU is contained
        //  within the bounds?
        bool is_linear = s_curve_l > 0;
        bool is_exponential = s_curve_u < 0;
        if (is_linear) {
            *slope_inf = 1;
            *slope_sup = 1;
        }
        else if (is_exponential) {
            // Since elu(x) = e^x - 1, then elu'(x) = e^x = elu(x) + 1
            *slope_inf = s_curve_l + 1;
            *slope_sup = s_curve_u + 1;
        }
        else {
            // When in both the linear and exponential regions, the exponential derivative will always be <= the linear
            // derivative so we can just default to the exponential derivative for the lower bound of the slope.
            // Similarly, we can default to the linear derivative for the upper bound of the slope.
            *slope_inf = s_curve_l + 1;
            *slope_sup = 1;
        }
    }
}

expr_t * create_s_curve_expr(fppoly_internal_t *pr, neuron_t *out_neuron, neuron_t *in_neuron, size_t i, bool is_lower, enum s_curve_type s_curve, bool use_default_heuristic){
	expr_t * res = alloc_expr();
	res->inf_coeff = (double *)malloc(sizeof(double));
	res->sup_coeff = (double *)malloc(sizeof(double));
	res->inf_cst = 0.0;
	res->sup_cst = 0.0;
	res->type = SPARSE;
	res->size = 1;
	res->dim = (size_t*)malloc(sizeof(size_t));
	res->dim[0] = i;
	double lb = in_neuron->lb;
	double ub = in_neuron->ub;

	double slope_inf, slope_sup;
	double intercept_inf, intercept_sup;
	fesetround(FE_DOWNWARD);
        double e_sup_l = s_curve == sigmoid ? -exp(ub) :
                        (s_curve == tanh_curve ? -tanh(ub) : -elu(ub));
        double e_inf_l = s_curve == sigmoid ? -exp(-lb) :
                        (s_curve == tanh_curve ? -tanh(-lb) : -elu(-lb));

        fesetround(FE_UPWARD);
        double e_sup_u = s_curve == sigmoid ? exp(ub) :
                        (s_curve == tanh_curve ? tanh(ub) : elu(ub));
        double e_inf_u = s_curve == sigmoid ? exp(-lb) :
                        (s_curve == tanh_curve ? tanh(-lb) : elu(-lb));

        double f_sup_l, f_sup_u;
        double f_inf_l, f_inf_u;
        double den_sup_l, den_sup_u;
        double den_inf_l, den_inf_u;
        double connecting_slope_l, connecting_slope_u;
        bool boxify = false;

        if(s_curve == sigmoid){
            den_sup_l = -1 + e_sup_l;
            den_sup_u = 1 + e_sup_u;
            den_inf_l = -1 + e_inf_l;
            den_inf_u = 1 + e_inf_u;
            elina_double_interval_div(&f_sup_l, &f_sup_u, e_sup_l, e_sup_u, den_sup_l, den_sup_u);
            elina_double_interval_div(&f_inf_l, &f_inf_u, e_inf_l, e_inf_u, den_inf_l, den_inf_u);
        }
        // Since we introduce and use the `elu` function, we can use the same approach as tanh (which uses the
        // `tanh` function) when computing the infimum and supremum values.
        else if(s_curve == tanh_curve || s_curve == elu_curve) {
            f_inf_l = e_inf_l;
            f_inf_u = e_inf_u;
            f_sup_l = e_sup_l;
            f_sup_u = e_sup_u;
            den_inf_l = e_inf_l;
            den_inf_u = e_inf_u;
            den_sup_l = e_sup_l;
            den_sup_u = e_sup_u;
        }

        out_neuron->lb = f_inf_l;
        out_neuron->ub = f_sup_u;
        if((-lb==ub) || (-f_inf_l==f_sup_u)){
	    slope_inf = 0.0;
	    slope_sup = 0.0;
	    intercept_inf = f_inf_l;
	    intercept_sup = f_sup_u;

        }

        double add_inf, add_sup;
        double mul_inf, mul_sup;
        double x_l, x_u;
        double f_x_l, f_x_u;

        if(is_lower){
                if(ub<0){
                    compute_derivative(&slope_inf, &slope_sup, e_inf_l, e_inf_u, den_inf_l, den_inf_u, s_curve);
                    x_l = -lb;
                    x_u = lb;
                    f_x_l = f_inf_l;
                    f_x_u = f_inf_u;
                    elina_double_interval_mul_cst_coeff(pr,&add_inf,&add_sup,x_l, x_u, slope_inf,slope_sup);
                    elina_double_interval_add_cst_coeff(pr, &intercept_inf, &intercept_sup,f_x_l, f_x_u, add_inf, add_sup);
		    double tmp1, tmp2, tmp3, tmp4;
                    elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,-ub, ub, slope_inf, slope_sup);
                    elina_double_interval_add_cst_coeff(pr,&tmp3,&tmp4,intercept_inf, intercept_sup, tmp1, tmp2);
                    if(tmp4>-f_sup_l){
                        boxify = true;
                    }
                }
                else if(lb<=0){

		            compute_chord_slope(&slope_inf, &slope_sup, f_sup_l, f_sup_u, f_inf_l, f_inf_u, lb, -lb, -ub, ub);

			    x_l = -lb;
			    x_u = lb;
			    f_x_l = f_inf_l;
			    f_x_u = f_inf_u;
			    elina_double_interval_mul_cst_coeff(pr,&add_inf,&add_sup,x_l, x_u, slope_inf, slope_sup);
			    elina_double_interval_add_cst_coeff(pr, &intercept_inf, &intercept_sup,f_x_l, f_x_u, add_inf, add_sup);
			    double tmp1, tmp2, tmp3, tmp4;
                    	    elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,-ub, ub, slope_inf, slope_sup);
                            elina_double_interval_add_cst_coeff(pr,&tmp3,&tmp4, intercept_inf, intercept_sup, tmp1, tmp2);
                    	    if(-tmp3>f_sup_u || slope_sup+slope_inf>1e-5){
                        	boxify = true;
                    	    }
		            //}
                }
                else{

                    if(lb<=ub){
                    	  if(use_default_heuristic){
                         	compute_derivative(&slope_inf, &slope_sup, e_sup_l, e_sup_u, den_sup_l, den_sup_u, s_curve);
                         }
                         else{
                         	double slope_inf1, slope_sup1;
                         	double slope_inf2, slope_sup2;
                         	compute_derivative(&slope_inf1, &slope_sup1, e_inf_l, e_inf_u, den_inf_l, den_inf_u, s_curve);
                         	compute_chord_slope(&slope_inf2, &slope_sup2, f_sup_l, f_sup_u, f_inf_l, f_inf_u, lb, -lb, -ub, ub);
                         	if(slope_sup1<slope_sup2){
                         		slope_inf = slope_inf1;
                         		slope_sup = slope_sup1;
                         	}
                         	else{
                         		slope_inf = slope_inf2;
                         		slope_sup = slope_sup2;
                         	}
                         }
                    }
                    else{

                         compute_derivative(&slope_inf, &slope_sup, e_inf_l, e_inf_u, den_inf_l, den_inf_u, s_curve);

                    }

                    x_l = -lb;
                    x_u = lb;
                    f_x_l = f_inf_l;
                    f_x_u = f_inf_u;
                    elina_double_interval_mul_cst_coeff(pr,&add_inf,&add_sup,x_l, x_u, slope_inf, slope_sup);
                    elina_double_interval_add_cst_coeff(pr, &intercept_inf, &intercept_sup,f_x_l, f_x_u, add_inf, add_sup);
                    double tmp1, tmp2, tmp3, tmp4;
                    elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,-ub, ub, slope_inf, slope_sup);
                    elina_double_interval_add_cst_coeff(pr,&tmp3,&tmp4, intercept_inf, intercept_sup, tmp1, tmp2);
                    if(tmp4>-f_sup_l){
                        boxify = true;
                    }
                }

            }
       else{
                if(ub < 0){

		     compute_chord_slope(&slope_inf, &slope_sup, f_sup_l, f_sup_u, f_inf_l, f_inf_u, lb, -lb, -ub, ub);

		     x_l = ub;
		     x_u = -ub;
		     f_x_l = f_sup_l;
		     f_x_u = f_sup_u;
		     elina_double_interval_mul_cst_coeff(pr,&add_inf,&add_sup,x_l, x_u, slope_inf, slope_sup);
		     elina_double_interval_add_cst_coeff(pr, &intercept_inf, &intercept_sup,f_x_l, f_x_u, add_inf, add_sup);
		     double tmp1, tmp2, tmp3, tmp4;
                     elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,lb, -lb, slope_inf, slope_sup);
                     elina_double_interval_add_cst_coeff(pr,&tmp3,&tmp4, intercept_inf, intercept_sup, tmp1, tmp2);

                    if(tmp4<f_inf_u || slope_sup+slope_inf>1e-5){
                        boxify = true;
                    }
                }
                else if(lb<=0){

                    compute_derivative(&slope_inf, &slope_sup, e_sup_l, e_sup_u, den_sup_l, den_sup_u, s_curve);

                    x_l = ub;
                    x_u = -ub;
                    f_x_l = f_sup_l;
                    f_x_u = f_sup_u;
                    elina_double_interval_mul_cst_coeff(pr,&add_inf,&add_sup,x_l, x_u, slope_inf, slope_sup);
                    elina_double_interval_add_cst_coeff(pr, &intercept_inf, &intercept_sup,f_x_l, f_x_u, add_inf, add_sup);
                    double tmp1, tmp2, tmp3, tmp4;
                    elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,lb, -lb, slope_inf, slope_sup);
                    elina_double_interval_add_cst_coeff(pr,&tmp3,&tmp4, intercept_inf, intercept_sup, tmp1, tmp2);

                    if(-tmp3<f_inf_u){
                        boxify = true;
                    }


                }
                else{

                    if(lb<=ub){
                    	compute_derivative(&slope_inf, &slope_sup, e_sup_l, e_sup_u, den_sup_l, den_sup_u, s_curve);
                    }
                    else{
                    	if(use_default_heuristic){
                        	compute_derivative(&slope_inf, &slope_sup, e_inf_l, e_inf_u, den_inf_l, den_inf_u, s_curve);
                       }
                       else{
                         double slope_inf1, slope_sup1;
                         double slope_inf2, slope_sup2;
                         compute_derivative(&slope_inf1, &slope_sup1, e_sup_l, e_sup_u, den_sup_l, den_sup_u, s_curve);
                         compute_chord_slope(&slope_inf2, &slope_sup2, f_sup_l, f_sup_u, f_inf_l, f_inf_u, lb, -lb, -ub, ub);
                         if(slope_sup1<slope_sup2){
                         	slope_inf = slope_inf1;
                         	slope_sup = slope_sup1;
                         }
                         else{
                         	slope_inf = slope_inf2;
                         	slope_sup = slope_sup2;
                         }
                      }
                    }


                    x_l = ub;
                    x_u = -ub;
                    f_x_l = f_sup_l;
                    f_x_u = f_sup_u;
                    elina_double_interval_mul_cst_coeff(pr,&add_inf,&add_sup,x_l, x_u, slope_inf, slope_sup);
                    elina_double_interval_add_cst_coeff(pr, &intercept_inf, &intercept_sup,f_x_l, f_x_u, add_inf, add_sup);
                    double tmp1, tmp2, tmp3, tmp4;
                    elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,lb, -lb, slope_inf, slope_sup);
                    elina_double_interval_add_cst_coeff(pr,&tmp3,&tmp4, intercept_inf, intercept_sup, tmp1, tmp2);
                    if(-tmp3<f_inf_u){
                        boxify = true;
                    }

                }


       }
       if(boxify){
            slope_inf = 0.0;
	    slope_sup = 0.0;
	    if(is_lower){
	    	intercept_inf = f_inf_l;
	    	intercept_sup = -f_inf_l;
	    }
	    else{
	    	intercept_inf = -f_sup_u;
	    	intercept_sup = f_sup_u;
	    }

        }
	res->inf_coeff[0] = slope_inf;
	res->sup_coeff[0] = slope_sup;
	res->inf_cst = intercept_inf;
	res->sup_cst = intercept_sup;
	return res;
}



void handle_s_curve_layer(elina_manager_t *man, elina_abstract0_t* element, size_t num_neurons, size_t *predecessors, size_t num_predecessors, enum s_curve_type s_curve, bool use_default_heuristic){
	assert(num_predecessors==1);
	fppoly_t *fp = fppoly_of_abstract0(element);
	fppoly_internal_t *pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	size_t numlayers = fp->numlayers;
	fppoly_add_new_layer(fp, num_neurons, predecessors, num_predecessors, true);
	neuron_t **out_neurons = fp->layers[numlayers]->neurons;
	int k = predecessors[0]-1;
	neuron_t **in_neurons = fp->layers[k]->neurons;
	size_t i;
	for(i=0; i < num_neurons; i++) {
        out_neurons[i]->lexpr = create_s_curve_expr(pr, out_neurons[i], in_neurons[i], i, true, s_curve, use_default_heuristic);
        out_neurons[i]->uexpr = create_s_curve_expr(pr, out_neurons[i], in_neurons[i], i, false, s_curve, use_default_heuristic);
	}
}

void handle_sigmoid_layer(elina_manager_t *man, elina_abstract0_t* element, size_t num_neurons, size_t *predecessors, size_t num_predecessors, bool use_default_heuristic){
	handle_s_curve_layer(man, element, num_neurons, predecessors, num_predecessors, true, use_default_heuristic);
}

void handle_tanh_layer(elina_manager_t *man, elina_abstract0_t* element, size_t num_neurons, size_t *predecessors, size_t num_predecessors, bool use_default_heuristic){
	handle_s_curve_layer(man, element, num_neurons, predecessors, num_predecessors, false, use_default_heuristic);
}

void handle_elu_layer(elina_manager_t *man, elina_abstract0_t* element, size_t num_neurons, size_t *predecessors, size_t num_predecessors, bool use_default_heuristic){
	handle_s_curve_layer(man, element, num_neurons, predecessors, num_predecessors, false, use_default_heuristic);
}


double apply_s_curve_expr(fppoly_internal_t *pr, expr_t **uexpr_p, neuron_t * neuron, bool is_lower, enum s_curve_type s_curve){
	expr_t * uexpr = *uexpr_p;
	size_t i;
	size_t size = uexpr->size;
	double lb = neuron->lb;
	double ub = neuron->ub;

	neuron_t *tmp_neuron = neuron_alloc();
	expr_t * res = create_s_curve_expr(pr,tmp_neuron, neuron, 0, is_lower, s_curve, true);
	bool boxify = false;
	double slope_inf, slope_sup;
	double intercept_inf, intercept_sup;

	slope_inf = res->inf_coeff[0];
	slope_sup = res->sup_coeff[0];
	intercept_inf = res->inf_cst;
	intercept_sup = res->sup_cst;
	fesetround(FE_DOWNWARD);
	double e_sup_l = s_curve == sigmoid ? -exp(ub) :
                        (s_curve == tanh_curve ? -tanh(ub) : -elu(ub));
	fesetround(FE_UPWARD);
	double e_sup_u = s_curve == sigmoid ? exp(ub) :
                        (s_curve == tanh_curve ? tanh(ub) : elu(ub));
	double f_sup_l, f_sup_u;
	double den_sup_l, den_sup_u;
	if(s_curve == sigmoid){
		den_sup_l = -1 + e_sup_l;
		den_sup_u = 1 + e_sup_u;
		elina_double_interval_div(&f_sup_l, &f_sup_u, e_sup_l, e_sup_u, den_sup_l, den_sup_u);
	}
    // Since we introduce and use the `elu` function, we can use the same approach as tanh (which uses the `tanh`
    // function) when computing the infimum and supremum values.
	else if(s_curve == tanh_curve || s_curve == elu_curve){
		f_sup_l = e_sup_l;
		f_sup_u = e_sup_u;
	}

	for(i=0; i < size; i++){
		elina_double_interval_mul_expr_coeff(pr,&uexpr->inf_coeff[i],&uexpr->sup_coeff[i],slope_inf,slope_sup,uexpr->inf_coeff[i],uexpr->sup_coeff[i]);
	}
	elina_double_interval_mul_cst_coeff(pr, &uexpr->inf_cst, &uexpr->sup_cst, slope_inf, slope_sup, uexpr->inf_cst, uexpr->sup_cst );
	elina_double_interval_add_cst_coeff(pr,&uexpr->inf_cst,&uexpr->sup_cst,intercept_inf, intercept_sup, uexpr->inf_cst, uexpr->sup_cst);
	free_expr(res);
	free_neuron(tmp_neuron);
	return f_sup_u;
}

double apply_sigmoid_lexpr(fppoly_internal_t *pr, expr_t **lexpr_p, neuron_t * neuron){
	return apply_s_curve_expr(pr,lexpr_p,neuron,true,true);
}

double apply_tanh_lexpr(fppoly_internal_t *pr, expr_t **lexpr_p, neuron_t * neuron){
	return apply_s_curve_expr(pr,lexpr_p,neuron,true,false);
}

double apply_sigmoid_uexpr(fppoly_internal_t *pr, expr_t **uexpr_p, neuron_t * neuron){
	return apply_s_curve_expr(pr,uexpr_p,neuron,false,true);
}

double apply_tanh_uexpr(fppoly_internal_t *pr, expr_t **uexpr_p, neuron_t * neuron){
	return apply_s_curve_expr(pr,uexpr_p,neuron,false,false);
}





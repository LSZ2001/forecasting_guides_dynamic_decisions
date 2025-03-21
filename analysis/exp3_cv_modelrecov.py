
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_folds", type=int, default=5, help="CV number of folds")
parser.add_argument("--cv_seed", type=int, default=0, help="Random seed to partition CV training datasets")
parser.add_argument("--fakedataset_modelfullname_combidx", type=int, default=0, help="Which fakedataset and model to fit on it")
args = parser.parse_args()

# Dataset extraction
n_folds = args.n_folds
cv_seed = args.cv_seed
fakedataset_modelfullname_combidx = args.fakedataset_modelfullname_combidx


def exp3_cv_modelrecov(n_folds=5, cv_seed=0, fakedataset_modelfullname_combidx = 0):

    import numpy as np
    import scipy 
    import random
    import os
    import pickle
    import glob 
    import timeit
    from itertools import product
    # Use BADS
    import concurrent.futures
    from pybads import BADS

    data_path = os.path.join("learning_curves","data")
    fits_path = os.path.join("learning_curves","modelfits")

    print("Code started running...")



    #--------------------




    # 1) Functions for model fitting
    logistic_growth = lambda t, theta: theta[:,0] + theta[:,1]/(1+np.exp(-theta[:,2]*(t-theta[:,3])))
    dt=1  #1
    ts = np.arange(0,11,dt)[:,None]
    T_end = len(ts)-1;

    # Group-level parameters for each family
    theta_mean = 0.55
    theta_nu = 0.15
    condition_left_offsets = np.array([17-25/24, 3]) # Offsets for the seplate and sepearly conditions.

    def logistic_growth_group(ts, traj_assymtotes, left_offset):
        '''Here, theta is a num_thetas x 1 vector, 
        corresponding to theta[:,1] required by the lambda function logistic_growth().'''
        num_thetas,_ = traj_assymtotes.shape
        new_theta = np.zeros((num_thetas,4))
        new_theta[:,0] = 0 # Starting height
        new_theta[:,1] = np.squeeze(traj_assymtotes) # Asymptotic height
        new_theta[:,2] = -1/2*np.squeeze(traj_assymtotes) + 9/8 # Slope
        new_theta[:,3] = left_offset + 25/6*np.squeeze(traj_assymtotes) # Left/right translation
        return logistic_growth(3*ts, new_theta)

    def log_normpdf(x, mean, std):
        log_normpdf_unnorm = -np.log(std)-0.5*np.log(2*np.pi)-0.5*((x-mean)/std)**2
        return log_normpdf_unnorm

    ## Grid of integration. Needs to be more finer. 
    theta0_grid = np.linspace(0,1,50)  
    r_T_grid = np.linspace(-3,5,400)

    def traj_r_T_estimation(ts, rs_traj, sensory_noise=0.1, cond=0, t_now_idx=0, T_end=T_end):
        ts_train = ts[0:(t_now_idx+1)];

        # 1) Find p(theta|r) as a function of theta.
        log_p_thetaj_given_rj = log_normpdf(theta0_grid,theta_mean,theta_nu) # This is actually the log prior p(theta); [theta0_grid]
        rho_t_given_thetaj = logistic_growth_group(ts_train, theta0_grid[:,None], condition_left_offsets[cond]) # traj heights over time, given theta [ts_train, theta0_grid]
        if(t_now_idx>-1):
            # Update the prior p(theta) using the likelihoods over time p(r_{1:t} | theta), to obtain the posterior p(theta | r_{1:t})
            log_p_thetaj_given_rj = log_p_thetaj_given_rj + np.sum(log_normpdf(rho_t_given_thetaj, rs_traj[0:len(ts_train)][:,None], sensory_noise), axis=0) # np.sum is summing across time over log likelihoods
        # logsumexp
        log_p_thetaj_given_rj_normalized = log_p_thetaj_given_rj - scipy.special.logsumexp(log_p_thetaj_given_rj)


        # 2) Find p(r_T | rho(T; theta))
        rho_T_given_thetaj = logistic_growth_group(T_end, theta0_grid[:,None], condition_left_offsets[cond]) #[theta0_grid, ]
        log_r_T_given_thetaj = log_normpdf(r_T_grid[:,None], rho_T_given_thetaj[None,:], sensory_noise) # [r_T_grid, theta0_grid]

        # 3) Find p(r_T | r)
        log_p_r_T_and_theta_given_r = log_r_T_given_thetaj + log_p_thetaj_given_rj_normalized[None,:]
        p_r_T_given_r = np.trapz(np.exp(log_p_r_T_and_theta_given_r), x=theta0_grid, axis=-1)
        p_r_T_given_r = p_r_T_given_r/np.trapz(p_r_T_given_r, x=r_T_grid)
        post_mean = np.trapz(r_T_grid*p_r_T_given_r, x=r_T_grid)
        return post_mean

    p_stay_logistic = lambda diff, slope, offset: 1 / (1 + np.exp(offset - slope * diff))

    def prob_stay_logistic(theta, cond=0, ts=ts, t_now=0, T_end=T_end, traj=np.nan, model="bayesian_postmean", include_offset_param = True, return_postmean=False):
        '''Given a set of parameter values theta=(sensory_noise, slope, offset), 
            evaluate p(stay; theta, traj[0:(t_now+1)]) given the starting trajectory's development up until t_now, 
            under a certain model.'''
        sensory_noise = theta[0]
        slope = theta[1]
        if(include_offset_param):
            offset = theta[2]
        else:
            offset = 0
            
        if(model=="bayesian_postmean"):
            stay_finalheight_postmean = traj_r_T_estimation(ts, traj, sensory_noise=sensory_noise, cond=cond, t_now_idx=t_now, T_end=T_end)
            switch_finalheight_priormean = traj_r_T_estimation(ts, np.nan, sensory_noise=sensory_noise, cond=cond, t_now_idx=-1, T_end=ts[T_end-t_now][0])
            final_height_diff = stay_finalheight_postmean - switch_finalheight_priormean
            prob_stay = p_stay_logistic(final_height_diff, slope, offset)
            if(return_postmean):
                return (prob_stay, final_height_diff)

        elif(model=="velocity"):
            stay_finalheight_postmean = traj_r_T_estimation(ts, traj, sensory_noise=sensory_noise, cond=cond, t_now_idx=t_now, T_end=T_end)
            switch_finalheight_priormean = traj_r_T_estimation(ts, np.nan, sensory_noise=sensory_noise, cond=cond, t_now_idx=-1, T_end=ts[T_end-t_now][0])  
            stay_initial_height_postmean = traj_r_T_estimation(ts, traj, sensory_noise=sensory_noise, cond=cond, t_now_idx=t_now, T_end=t_now)
            switch_initial_height_priormean = traj_r_T_estimation(ts, np.nan, sensory_noise=sensory_noise, cond=cond, t_now_idx=-1, T_end=0)

            stay_velocity = (stay_finalheight_postmean - stay_initial_height_postmean)/(T_end-t_now)
            switch_velocity = (switch_finalheight_priormean - switch_initial_height_priormean)/(T_end-t_now)
            velocity_diff = stay_velocity - switch_velocity
            prob_stay = p_stay_logistic(velocity_diff, slope, offset)
            if(return_postmean):
                return (prob_stay, velocity_diff)

        elif(model=="exploitation_heuristic"):
    #         stay_currentheight = traj[t_now]
    #         switch_initial_height_priormean = traj_r_T_estimation(ts, np.nan, sensory_noise=sensory_noise, cond=cond, t_now_idx=-1, T_end=0)
    #         current_height_diff = stay_currentheight - switch_initial_height_priormean
    #         prob_stay = p_stay_logistic(current_height_diff, slope, offset)
            prob_stay = p_stay_logistic(traj[t_now], slope, offset)
            if(return_postmean):
                return (prob_stay, traj[t_now])

        elif(model=="velocity_heuristic"):
            velocity_diff = (traj[t_now]- traj[t_now-1]) - (traj[t_now]-traj[0])/(t_now)
            prob_stay = p_stay_logistic(velocity_diff, slope, offset)
            if(return_postmean):
                return (prob_stay, velocity_diff)



        return prob_stay



    T_ends = [10,10]

    def crossentropy_loss(theta, data_subj=np.nan, ts=ts, T_ends=T_ends, model="bayesian_postmean",include_offset_param=True, cross_validate=False, cross_validate_timepts=np.nan):
        '''For the data of one subject: [cond, run, t_now, (traj_height, response)],
        and a set of parameter values theta, compute cross entropy loss.'''

        loss = 0;
        for cond in range(n_conds):
            T_end = T_ends[cond]
            for run in range(n_runs):
                traj = data_subj[cond,run,:,0]
                responses = data_subj[cond,run,:,1] # 1 is stay; 0 is switch
                n_steps_beforeswitch = np.sum(1-np.isnan(responses))
                for t_now_idx in range(1,n_steps_beforeswitch+1): # Exclude Month 0, and any month after the switch (if switched)

                    if(cross_validate): # Doing cross-validation: only evaluate loss on a subset of dpts.
                        target_tuple = (cond,run,t_now_idx)

                        if(target_tuple in cross_validate_timepts):
                            model_prob_pred = prob_stay_logistic(theta, cond=cond, traj=traj, ts=ts, t_now=ts[t_now_idx][0], T_end=T_end, model=model,include_offset_param=include_offset_param)
                            response = responses[t_now_idx]
                            if(response==1):
                                loss = loss - np.log(max(model_prob_pred,1e-100))
                            elif(response==0):
                                loss = loss - np.log(max(1-model_prob_pred,1e-100))
                            else:
                                print("Warning: trials include nan!")
                        else: 
                            continue;

                    else: # Not doing cross-validation: fitted to the full dataset.
                        model_prob_pred = prob_stay_logistic(theta, cond=cond, traj=traj, ts=ts, t_now=ts[t_now_idx][0], T_end=T_end, model=model,include_offset_param=include_offset_param)
                        response = responses[t_now_idx]
                        if(response==1):
                            loss = loss - np.log(max(model_prob_pred,1e-100))
                        elif(response==0):
                            loss = loss - np.log(max(1-model_prob_pred,1e-100))
                        else:
                            print("Warning: trials include nan!")
        return loss


    def evaluate_p_stay_logistic(theta, data_subj=np.nan, ts=ts, T_ends=T_ends, model="bayesian_postmean",include_offset_param=True):   
        loss = 0;
        model_prob_preds = np.zeros((n_conds, n_runs, n_timesteps))
        model_prob_preds[:] = np.nan;
        stayswitch_diffs = np.zeros((n_conds, n_runs, n_timesteps))
        stayswitch_diffs[:] = np.nan;
        for cond in range(n_conds):
            T_end = T_ends[cond]
            for run in range(n_runs):
                traj = data_subj[cond,run,:,0]
                responses = data_subj[cond,run,:,1] # 1 is stay; 0 is switch
                n_steps_beforeswitch = np.sum(1-np.isnan(responses))
                for t_now_idx in range(1,n_steps_beforeswitch+1): # Exclude Month 0, and any month after the switch (if switched)
                    [model_prob_pred, stayswitch_diff] = prob_stay_logistic(theta, cond=cond, traj=traj, ts=ts, t_now=ts[t_now_idx][0], T_end=T_end, model=model, return_postmean=True,include_offset_param=include_offset_param)
                    model_prob_preds[cond, run, t_now_idx] = model_prob_pred
                    stayswitch_diffs[cond, run, t_now_idx] = stayswitch_diff
        return (model_prob_preds, stayswitch_diffs)


    def cross_validation_dataset_partition(responses_allsubj, n_folds=5, seed=0):
        # Cross validation dataset partitioning?
        n_subj = responses_allsubj.shape[1] # [conds, subj, runs, timepts]
        cross_val_setidxs_allsubj = [] # Nested array: subjs -> folds -> (train dataset, test dataset) -> [cond,run,timept] signatures of a trial.

        for subj in range(n_subj):
            responses = responses_allsubj[:,subj,:,:]

            # Step 1: Identify valid trials (0 or 1)
            valid_indices = np.argwhere((responses == 0) | (responses == 1))

            # Step 2: Shuffle indices
            np.random.seed(seed)
            np.random.shuffle(valid_indices)

            # Step 3: Split into 5 folds
            folds = np.array_split(valid_indices, n_folds)

            # Step 4: Create training and testing trial idxs for each fold
            cross_val_setidxs = []
            for i in range(n_folds):
                test_indices = folds[i]
                train_indices = np.concatenate(folds[:i] + folds[i+1:])
                train_indices = [tuple(idx) for idx in train_indices]
                test_indices = [tuple(idx) for idx in test_indices]
                cross_val_setidxs.append((train_indices, test_indices)) # First level are the 5 folds; second level is a tuple with [0] being the training dataset.
            cross_val_setidxs_allsubj.append(cross_val_setidxs)

        return cross_val_setidxs_allsubj






    #----------------------





    # 2) Load the fake datasets for all models
    filename = os.path.join(data_path, "exp3_fits_fakedata_allmodels")
    with open(filename+'.pickle', 'rb') as handle:
        b = pickle.load(handle)
    models_true_gen_params = b['true_gen_params']
    models_fakedata = b['models_fakedata']
    [n_conds, n_subj, n_runs, n_timepts, _] = models_fakedata["bayesian_postmean_offset"].shape


    # CV dataset partition: Loop over fake datasets to fit models on.
    cross_val_setidxs_allsubjs_fakedata = {}
    for fake_dataset_genmodel in models_fakedata.keys():
        fake_dataset = models_fakedata[fake_dataset_genmodel]
        cross_val_setidxs_allsubjs_fakedata[fake_dataset_genmodel] = cross_validation_dataset_partition(fake_dataset[:,:,:,:,1], n_folds, cv_seed)
    # Print length of train and test dataset for a particular subj's particular fold.
    print("Fake datasets loaded and CV-partitioned: n_folds="+str(n_folds)+", seed="+str(cv_seed))
    
    
    # Get fakedataset_modelfullname_pairs.
    model_fullnames = [key for key in models_fakedata.keys()]
    # Convert to JSON-formatted string with double quotes
    #model_fullnames = json.dumps(model_fullnames, indent=2)
    fakedataset_modelfullname_pairs = list(product(model_fullnames, repeat=2))
    
    
    # Get mTurk IDs
    filename = os.path.join(data_path,"exp3_data")
    with open(filename+'.pickle', 'rb') as handle:
        b = pickle.load(handle)
    mturk_IDs = b["mturk_IDs"]






    #-------------------------





    # 3) CV model recovery
    # The fake datasets used to perform model fits on
    # Models to be fitted
    fake_dataset_genmodels, model_fullname = fakedataset_modelfullname_pairs[fakedataset_modelfullname_combidx]
    include_offset_param_vals = "_offset" in model_fullname
    models = [model_fullname.rpartition('_')[0]]
    n_inits = 10
    n_foldsinits = n_inits*n_folds

    print()
    print("Fake datasets: ", fake_dataset_genmodels)
    print("Model: ", models)
    print("Include_offset_param_val: ", include_offset_param_vals)
    print()

    # Lopp over fake datasets to fit models on.
    for fake_dataset_genmodel in [fake_dataset_genmodels]:
        fake_dataset = models_fakedata[fake_dataset_genmodel]
        true_gen_params = models_true_gen_params[fake_dataset_genmodel]
        cross_val_setidxs_allsubjs = cross_val_setidxs_allsubjs_fakedata[fake_dataset_genmodel]

        # Loop over models to be fitted
        for include_offset_param in [include_offset_param_vals]:
            for model_idx in [0]:
                model = models[model_idx]
                if(include_offset_param):
                    lb = [1e-3, 1e-3, -10]
                    ub = [1.5, 50, 10]
                    plb = [0.01,10,-2]
                    pub = [0.5, 40, 2]
                else:
                    lb = [1e-3, 1e-3]
                    ub = [1.5, 50]
                    plb = [0.01,10]
                    pub = [0.5, 40]

                constraint = scipy.optimize.Bounds(lb=lb, ub=ub)
                params_init = np.zeros((n_subj, n_folds, n_inits, len(lb)))
                params_fitted = np.zeros((n_subj, n_folds, n_inits, len(lb)))
                crossentropy_fitted = np.zeros((n_subj, n_folds, n_inits))
                total_time = np.zeros((n_subj, n_folds, n_inits))


                # Loop over subjects
                for subj in range(n_subj): # Just Subject 0.
                    data_subj = fake_dataset[:,subj,:,:,:]
                    cross_val_setidxs = cross_val_setidxs_allsubjs[subj]
                    print("\n-------------")

                    def task(init):
                        fold = init//n_inits #iter 0 to 9 are for fold 0, 10 to 19 are for fold 1, etc.
                        print("Dataset: Human", ", Model:", model_fullname, ", Subj",subj,", Fold", fold, ", Init",init%n_inits) 
                        target = lambda theta: crossentropy_loss(theta, data_subj=data_subj, ts=ts, T_ends=T_ends, model=model, include_offset_param=include_offset_param, cross_validate=True, cross_validate_timepts=cross_val_setidxs[fold][0])

                        # Do some computation here
                        params_init = np.random.uniform(low=plb, high=pub)
                        bads = BADS(target, params_init, lb, ub, plb, pub)
                        optimize_result = bads.optimize()
                        return optimize_result
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        # Submit tasks to the executor
                        futures = [executor.submit(task, i) for i in range(n_foldsinits)]
                        # Collect the results
                        results = [future.result() for future in concurrent.futures.as_completed(futures)]
                    #optimize_results.append(results)
                    #print(results)

                    for rand_init_idx in range(n_foldsinits):
                        params_init[subj, rand_init_idx//n_inits, rand_init_idx%n_inits,:] = results[rand_init_idx].x0
                        params_fitted[subj, rand_init_idx//n_inits, rand_init_idx%n_inits,:] = results[rand_init_idx].x
                        crossentropy_fitted[subj, rand_init_idx//n_inits, rand_init_idx%n_inits] = results[rand_init_idx].fval
                        total_time[subj, rand_init_idx//n_inits, rand_init_idx%n_inits] = results[rand_init_idx].total_time


                    filename = os.path.join(fits_path, "exp3_fits_cv_modelrecov_temp__d_"+fake_dataset_genmodel+"__m_"+model_fullname)
                    #filename = "exp3_fits_temp"
                    a = {'params_init':params_init,'params_fitted':params_fitted, 'cross_val_setidxs_allsubjs':cross_val_setidxs_allsubjs, 'crossentropy_fitted':crossentropy_fitted, 'model':model, 'model_include_offset_param':include_offset_param, 'model_fullname':model_fullname, 'total_time':total_time, 'mturk_IDs': mturk_IDs, 'data_dims': ["condition (0 is separate late)","subject","run","timestep (Month 0 to 9)","data (current trajectory height, current decision)"], 'fake_dataset_genmodel':fake_dataset_genmodel, 'true_gen_params':true_gen_params, 'fake_dataset': fake_dataset}
                    with open(filename+'.pickle', 'wb') as handle:
                        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # Find best set of fitted params across rand inits, for each model+offset comb.
                best_params_fitted = np.zeros((n_subj, len(models), n_folds, len(lb)))
                for subj in range(n_subj):
                    for model_idx in range(len(models)):
                        for fold in range(n_folds):
                            best_params_idx = np.argmin(crossentropy_fitted[subj, fold,:])
                            best_params_fitted[subj,model_idx,fold,:] = params_fitted[subj, fold,best_params_idx,:]

                # Save fitted model
                filename = os.path.join(fits_path, "exp3_fits_cv_modelrecov__d_"+fake_dataset_genmodel+"__m_"+model_fullname)
                a = {'params_init':params_init,'params_fitted':params_fitted, 'cross_val_setidxs_allsubjs':cross_val_setidxs_allsubjs, 'best_params_fitted':best_params_fitted, 'crossentropy_fitted':crossentropy_fitted, 'model':model, 'model_include_offset_param':include_offset_param, 'model_fullname':model_fullname, 'total_time':total_time, 'mturk_IDs': mturk_IDs, 'data_dims': ["condition (0 is separate late)","subject","run","timestep (Month 0 to 9)","data (current trajectory height, current decision)"], 'fake_dataset_genmodel':fake_dataset_genmodel, 'true_gen_params':true_gen_params, 'fake_dataset': fake_dataset}
                with open(filename+'.pickle', 'wb') as handle:
                    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("CV model fits have ended.")

exp3_cv_modelrecov(n_folds, cv_seed, fakedataset_modelfullname_combidx)
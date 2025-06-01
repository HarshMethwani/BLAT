import time
import pandas as pd
import torch
from tabulate import tabulate
from tqdm import tqdm
import datetime

def train_PPO_agent(
    env: object,
    agent: object,
    save_flag: int,
    epochs: int,
    total_episodes: int,
    seed: int,
    evaluator: object,
    **kwargs,
):
    actor_loss_list = []
    critic_loss_list = []
    return_list = []
    time_list = []
    seed_list = []
    llm_judge_list = []
    asr_judge_list = []
    intent_score_list = []
    soft_jb_step_list = []
    hard_jb_step_list = []

    start_time = time.time()
    best_score = -1e10

    val_interval = kwargs.get('val_interval', 5)
    val_num = kwargs.get('val_num', 2)
    val_max_step = kwargs.get('val_max_step', 10)
    
    # Log training start
    print(f"\n{'='*80}")
    print(f"🚀 Starting training at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Configuration: Epochs={epochs}, Episodes={total_episodes}, Seed={seed}")
    print(f"{'='*80}\n")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_return = 0
        epoch_success_count = 0
        
        # Print epoch header
        print(f"\n{'-'*30} EPOCH {epoch+1}/{epochs} {'-'*30}")
        
        # Use tqdm for progress tracking of episodes
        for episode in tqdm(range(total_episodes), desc=f"Epoch {epoch+1}", ncols=100):
            epi_training = False
            transition_dict = {}
            episode_return = 0

            # * ---- execute simulation ----
            soft_jb_step = -1
            hard_jb_step = -1
            llm_judge, asr_judge, intent_score = [], [], []

            state, info = env.reset(brandnew_train=True)
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, info = env.step(action)
                transition_dict = update_transition(epi_training, transition_dict, state,
                                                        done, action, next_state, reward)
                epi_training = True
                state = next_state
                episode_return += reward
                llm_judge += [info['llm_judge']]
                asr_judge += [info['asr_judge']]
                intent_score += [info['intent_score']]
                if info['s_jb_step'] > 0 and soft_jb_step == -1:
                    soft_jb_step = info['s_jb_step']
                if info['h_jb_step'] > 0 and hard_jb_step == -1:
                    hard_jb_step = info['h_jb_step']

            # Log successful jailbreaks if they occurred
            if 1 in asr_judge:
                epoch_success_count += 1
                tqdm.write(f"✅ Episode {episode+1}: Jailbreak successful at step {hard_jb_step}")
            
            # Update agent's policy
            actor_loss, critic_loss = agent.update(transition_dict)
            
            # Collect data for analysis
            return_list.append(episode_return)
            actor_loss_list.append(actor_loss)
            critic_loss_list.append(critic_loss)
            seed_list.append(seed)
            time_list.append(time.time() - start_time)
            llm_judge_list.append(sum(llm_judge) / len(llm_judge) if llm_judge else 0)
            asr_judge_list.append(sum(asr_judge) / len(asr_judge) if asr_judge else 0)
            intent_score_list.append(sum(intent_score) / len(intent_score) if intent_score else 0)
            soft_jb_step_list.append(soft_jb_step)
            hard_jb_step_list.append(hard_jb_step)
            
            epoch_return += episode_return

        # Run validation if needed
        if val_interval > 0 and (epoch + 1) % val_interval == 0:
            print(f"\n📝 Running validation after epoch {epoch+1}...")
            test_info, val_info_whole = env.validation(RL_agent=agent, test_num=val_num, max_step=val_max_step)
            evaluator.val_save(save_flag, seed, test_info, val_info_whole)
            
            score = test_info.get('success_rate', 0)
            if score > best_score:
                best_score = score
                evaluator.model_save(save_flag, seed, agent, best=True)
                print(f"🏆 New best model saved! Success rate: {score:.2%}")
        
        # End of epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_return = epoch_return / total_episodes if total_episodes > 0 else 0
        success_rate = epoch_success_count / total_episodes if total_episodes > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"📊 Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s")
        print(f"📈 Average return: {avg_return:.4f} | Success rate: {success_rate:.2%}")
        print(f"🧠 Actor loss: {actor_loss:.6f} | Critic loss: {critic_loss:.6f}")
        print(f"{'='*80}\n")
        
        # Save checkpoint after each epoch
        evaluator.model_save(save_flag, seed, agent, epoch=epoch+1)
        print(f"💾 Checkpoint saved for epoch {epoch+1}")

    # Training complete
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'='*80}")
    print(f"🎉 Training completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"📊 Final best success rate: {best_score:.2%}")
    print(f"{'='*80}\n")

    return return_list, time_list

def update_transition(epi_training, transition_dict, state, done, action, next_state, reward):
    '''
    Since the state and other data of each agent are stored separately in the form of dictionaries, they are merged here.
    '''
    for key, element in zip(['states', 'actions', 'next_states', 'rewards', 'dones'],
                            [state, action, next_state, reward, done]):
        if not epi_training:
            transition_dict[key] = torch.tensor(element).unsqueeze(0)
        else:
            transition_dict[key] = torch.cat([transition_dict[key], torch.tensor(element).unsqueeze(0)])
    return transition_dict
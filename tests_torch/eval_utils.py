import gymnasium as gym
from orangerl import AgentStage, EnvironmentStep, Agent, AgentOutput
from orangerl_torch import NNAgent
import tqdm

def eval_agent(
    eval_env : gym.Env,
    agent : NNAgent,
    num_episodes : int,
) -> float:
    agent.current_stage = AgentStage.EVAL
    total_reward = 0.0
    
    for i in range(num_episodes):
        observation, info = eval_env.reset()
        agent_last_state = None
        done = False
        while not done:
            agent_output = agent.get_action(observation, state = agent_last_state)
            action = agent_output.action.detach().cpu().numpy()
            agent_last_state = agent_output.state.detach() if agent_output.state is not None else None
            next_observation, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
            observation = next_observation
    return total_reward / num_episodes

def evaluate_training_performance(
    env: gym.Env,
    agent: NNAgent,
    steps : int,
    warmup_steps : int,
    eval_episodes : int,
    disable_tqdm : bool = False,
    enable_wandb : bool = False,
    wandb_project : str = "orangerl_test",
    wandb_run_name : str = "test_run",
    wandb_log_interval : int = 1000,
) -> float:
    if enable_wandb:
        import wandb
        wandb_run = wandb.init(project=wandb_project, name=wandb_run_name)

    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=0)
    agent.current_stage = AgentStage.ONLINE
    observation, info = env.reset()
    agent_last_state = None
    done = False
    update_info = None

    for i in tqdm.trange(steps, disable=disable_tqdm):
        agent_output = agent.get_action(observation, state = agent_last_state)
        action = agent_output.action.detach().cpu().numpy()
        agent_last_state = agent_output.state.detach() if agent_output.state is not None else None
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        transition = EnvironmentStep(
            observation=observation,
            next_observation=next_observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info = info
        )
        agent.observe_transitions(transition)
        
        if i > warmup_steps:
            update_info = agent.update()
        
        observation = next_observation
        if done:
            # print("Episode return: ", info["episode"]['r'])
            # print("Update Info", update_info)

            if enable_wandb:
                wandb.log({"episode_return": info["episode"]['r']}, step=i)

            observation, info = env.reset()
            done = False
            agent_last_state = None

        if i > warmup_steps and i % wandb_log_interval == 0 and enable_wandb:
            update_info_to_log = dict([
                ('agent/' + key, value) for key, value in update_info.items() if isinstance(value, (int, float))
            ])
            env_info_to_log = dict([
                ('env/' + key, value) for key, value in info.items() if isinstance(value, (int, float))
            ])
            wandb.log({**update_info_to_log, **env_info_to_log}, step=i)

    final_performance = eval_agent(env, agent, eval_episodes)
    return final_performance

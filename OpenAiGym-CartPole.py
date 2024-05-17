import gym 
import stable_baselines3
import stable_baselines3
from stable_baselines3.common.evaluation import evaluate_policy

class GymAgent():
    def __init__(self, env_name) -> None:
        self.env = None
        self.env_name = env_name
        self.actions = None
        self.state = None
        self.agent = None

    def build_agent(self):
        self.agent = stable_baselines3.PPO("MlpPolicy",self.env,verbose=1)

    def train_agent(self):
        self.agent.learn(total_timesteps=50000)

    def test_agent(self,num_episodes):
        evaluate_policy(self.agent, self.env, n_eval_episodes=num_episodes, render=True)

    def set_enviroment(self):
        self.env = gym.make(self.env_name)
        self.states = self.env.observation_space.shape[0]
        self.actions = self.env.action_space.n
        #print("State: {} Action:{}".format(self.states,self.actions))
        return
    
    def run_enviroment(self):
        done = False
        score = 0
        state = self.env.reset()
        while not done:
            self.env.render()
            action,_ = self.agent.predict(state)
            state, reward, done, info = self.env.step(action)
            score += reward 
        #print("Score {}".format(score))
        self.env.close()
        return
    
if __name__ == "__main__":
    agent = GymAgent("CartPole-v1")
    agent.set_enviroment()
    agent.build_agent()
    agent.train_agent()
    agent.test_agent(num_episodes=10)
    agent.run_enviroment()
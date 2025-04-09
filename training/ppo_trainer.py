import torch
from torch.optim import Adam

def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages

# PPO training function
def train_ppo(env, agent, epochs=1000, gamma=0.99, lam=0.95, clip_epsilon=0.2, lr=3e-4):
    optimizer = Adam(agent.parameters(), lr=lr)
    for epoch in range(epochs):
        obs, info = env.reset()
        done = False
        log_probs, values, rewards, states, actions = [], [], [], [], []

        # Data collection phase - detach from computation graph
        with torch.no_grad():
            while not done: 
                # Extract the image observation and flatten it
                state = torch.tensor(obs.flatten(), dtype=torch.float32)
                policy, value = agent(state)
                action = torch.multinomial(policy, 1).item()

                log_prob = torch.log(policy[action])
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                log_probs.append(log_prob)
                values.append(value.item())
                rewards.append(reward)
                states.append(state)
                actions.append(action)

        values.append(0)  # Bootstrap value for terminal state
        advantages = compute_advantages(rewards, values, gamma, lam)

        # Convert to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions)
        old_log_probs = torch.stack(log_probs).detach()  # Explicitly detach
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)

        # PPO update
        for _ in range(10):  # Number of policy updates per epoch
            # Get fresh policy and value estimates
            policy, value = agent(states)
            dist = torch.distributions.Categorical(policy)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - value.squeeze()).pow(2).mean()

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} completed")

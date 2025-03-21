# Interpretable and Steerable Sequence Learning via Prototypes (ProSeNet)

## 1. Settings

### 1.1 ProSeNet Architecture Overview

ProSeNet is designed for interpretable text classification by learning representative prototype sequences that serve as both classification references and analogous explanations. The overall architecture consists of three primary components:

- A **sequence encoder** $r$ that maps a variable-length input sequence into a fixed-length embedding.
- A **prototype layer** $p$ that holds $k$ prototype vectors.
- A **fully connected layer** $f$ (followed by a softmax layer) that uses the similarity scores between the embedding and the prototypes to output class probabilities.

The model is trained on a labeled sequence dataset 
$$
D = \{((x^{(t)})_{t=1}^{T}, y)\},
$$
where:
- $T$ is the sequence length,
- $x^{(t)} \in \mathbb{R}^{n}$ is the input vector at time step $t$,
- $y \in \{1,\dots,C\}$ is the label associated with the sequence.

The aim is to learn representative prototype sequences—possibly not present in the training data—that capture the essential semantic elements of the input.

### 1.2 Sequence Encoder

For a given input sequence $(x^{(t)})_{t=1}^{T}$, the sequence encoder $r$ transforms the entire sequence into a single embedding vector:
$$
e = r((x^{(t)})_{t=1}^{T}), \quad e \in \mathbb{R}^{m}.
$$
Key design details include:

- **Model Choice:**  
  The encoder can be instantiated with any backbone sequence learning model (e.g., LSTM, Bidirectional LSTM, or GRU). In the original experiments, the hidden state at the final time step, $h^{(T)}$, is used as the embedding $e$.

- **Design Assumption:**  
  The use of a recurrent encoder to collapse the sequential information into a single vector $e$ rests on the assumption that the final hidden state adequately encapsulates the temporal dynamics and salient features of the entire sequence.

### 1.3 Prototype Layer

The prototype layer $p$ contains $k$ prototype vectors:
$$
\{ p_i \}_{i=1}^{k}, \quad p_i \in \mathbb{R}^{m}.
$$
These prototypes are learned so that they represent prototypical sequences in the same latent space as the embedding $e$. The layer operates as follows:

- **Distance Computation:**  $d_i^2 = \| e - p_i \|_2^2$
- **Similarity Score:**  $a_i = \exp(-d_i^2)$
- **Interpretability Rationale:**  
	These scores indicate how closely an input sequence resembles each representative prototype, thereby providing an interpretable explanation for the classification.

### 1.4 Fully Connected Layer and Softmax

Once the similarity vector $a = [a_1, a_2, \dots, a_k]^\top$ is obtained from the prototype layer, the fully connected layer processes this information as follows:

- **Linear Transformation:**  
  The layer computes:
  $$
  z = W a,
  $$
  where $W \in \mathbb{R}^{C \times k}$ is a weight matrix (with $C$ being the number of classes). To further enhance interpretability, the weights in $W$ are constrained to be non-negative.

- **Probability Computation:**  
  Finally, a softmax layer is applied to convert the output $z$ into predicted probabilities for multi-class classification:
  $$
  \hat{y}_i = \frac{\exp(z_i)}{\sum_{j=1}^{C} \exp(z_j)}.
  $$

In summary, the “Settings” of ProSeNet are defined by a recurrent sequence encoder that distills variable-length input into a fixed-length vector, a prototype layer that leverages the squared $L_2$ distance and an exponential transformation to generate interpretable similarity scores, and a fully connected layer that aggregates these scores to yield a probabilistic classification.

---

## 2. Optimizing Objectives

### 2.1 Learning Objective

#### Cross-Entropy Loss

The cross-entropy (CE) loss is given by

$$
CE(\Theta, \mathcal{D}) = \sum_{(x^{(t)}, y) \in \mathcal{D}} \; y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}),
$$

where $\Theta$ denotes all trainable parameters of the model, $y$ is the ground truth label, and $\hat{y}$ is the predicted probability.

#### Diversity Regularization

To prevent the model from learning duplicate or highly similar prototypes—especially when the number of prototypes $k$ is large—a diversity regularization term is introduced. This term penalizes prototype pairs that are closer than a predefined threshold $d_{min}$:

$$
R_d(\Theta) = \sum_{i=1}^{k} \sum_{j=i+1}^{k} \max \left(0, d_{min} - \|p_i - p_j\|_2 \right)^2.
$$

Here, $p_i$ and $p_j$ are prototype vectors and $d_{min}$ is typically set to 1.0 or 2.0. This regularization encourages a more even spread of prototypes across the latent space and promotes a sparser similarity vector $\mathbf{a}$.

#### Sparsity and Non-Negativity

To further bolster interpretability, an $L_1$ penalty is applied to the fully connected layer $f$. In addition, the weight matrix $\mathbf{W}$ in this layer is constrained to be non-negative. The $L_1$ penalty enforces sparsity, ensuring that each prototype contributes in a unitary and additive manner to the final classification decision.

#### Clustering and Evidence Regularization

Two additional regularization terms are employed to align the latent representations with the learned prototypes:

- **Clustering Regularization $R_c$:** This term encourages the latent space to exhibit a clustering structure by minimizing the squared distance between each encoded instance and its closest prototype:

 $$
 R_c(\Theta, \mathcal{D}) = \sum_{(x_t^{(t)})_{t=1}^T \in \mathcal{X}} \min_{i=1}^k \left\| r\left((x_t^{(t)})_{t=1}^T\right) - p_i \right\|_2^2,
 $$

 where $r((x_t^{(t)})_{t=1}^T)$ is the sequence encoder’s output for a given input and $\mathcal{X}$ is the set of all sequences in the training set $\mathcal{D}$.

- **Evidence Regularization $R_e$:** This term forces each prototype vector to be as close as possible to some encoded instance, thereby ensuring that prototypes are grounded in actual training data:

 $$
 R_e(\Theta, \mathcal{D}) = \sum_{i=1}^k \min_{(x_t^{(t)})_{t=1}^T \in \mathcal{X}} \left\| p_i - r\left((x_t^{(t)})_{t=1}^T\right) \right\|_2^2.
 $$

#### Full Objective

By combining these components, the full loss function that ProSeNet minimizes is expressed as

$$
Loss(\Theta, \mathcal{D}) = CE(\Theta, \mathcal{D}) + \lambda_c\, R_c(\Theta, \mathcal{D}) + \lambda_e\, R_e(\Theta, \mathcal{D}) + \lambda_d\, R_d(\Theta, \mathcal{D}) + \lambda_{l_1}\, \|W\|_1,
$$

where $\lambda_c$, $\lambda_e$, $\lambda_d$, and $\lambda_{l_1}$ are hyperparameters controlling the strength of each regularization term. The selection of these hyperparameters depends on the nature of the data and is typically performed via cross-validation.

### 2.2 Optimization Strategy

#### Prototype Projection

A unique aspect of ProSeNet is the projection of prototype vectors onto the latent representations of observed training sequences. Because the prototypes lie in the latent space, they are not inherently interpretable. To address this, a projection step is performed periodically (e.g., every 4 training epochs):

$$
\mathbf{p}_i \gets \arg\min_{e \in r(X)} \| e - \mathbf{p}_i \|_2,
$$

where $r(X)$ denotes the set of latent embeddings for all sequences $X$ in the training set. This operation ensures that each prototype is associated with an actual observed sequence, thereby enhancing its interpretability.

#### Prototype Simplification

Even after projection, prototype sequences may contain extraneous or noisy information. To further simplify, the model projects each prototype onto an optimal subsequence that contains only the critical events. Specifically, the projection is modified as follows:

$$
\mathbf{p}_i \gets r(\text{seq}_i),
$$
$$
\text{seq}_i = \arg\min_{\text{seq} \in \text{sub}(\mathcal{X})} \left\| r(\text{seq}) - \mathbf{p}_i \right\|_2,
$$

where $\text{sub}(\mathcal{X})$ represents the set of all possible subsequences from the training set, and $|\cdot|$ computes the effective length of a subsequence. Given the exponential complexity $O(2^T N)$ of exhaustively searching over all subsequences (with $T$ being the maximum sequence length and $N$ the number of training sequences), a beam search heuristic (with beam width $w=3$) is employed to find an approximate solution with reduced computational cost ($O(w \cdot T^2 N)$).

---

In summary, the optimization objectives in ProSeNet are carefully formulated to balance accuracy with interpretability. The cross-entropy loss drives classification performance, while the clustering, evidence, diversity, and $L_1$ regularizations enforce a latent space structure where prototypes are both distinct and representative of meaningful subsequences. The optimization process, which includes SGD and periodic prototype projection (and simplification), guarantees that the learned prototypes are not only effective for prediction but also readily interpretable.


## 3. Interpretability

---

### 3.1 Prototype Visualization

Each prototype in ProSeNet is intrinsically linked to a concrete sequence (or subsequence) drawn from the training data, making the model’s decision process more transparent. The visualization process unfolds as follows:

- **Initial Prototype Association and Simplification:**  
  During training, the sequence encoder $r$ maps an input sequence $(x^{(t)})_{t=1}^{T}$ to a latent embedding $e$. Each prototype vector $\mathbf{p}_i$ (with $\mathbf{p}_i \in \mathbb{R}^m$) is learned in the same latent space. To render prototypes interpretable, a projection step assigns each $\mathbf{p}_i$ to its closest observed sequence embedding. For enhanced clarity, a prototype simplification procedure identifies a critical subsequence that captures the essence of the prototype. Formally, the simplified prototype is obtained by:
  
  $$
  \mathbf{p}_i \gets r(\text{seq}_i), \quad \text{where} \quad \text{seq}_i = \arg\min_{\text{seq} \in \text{sub}(\mathcal{X})} \|r(\text{seq}) - \mathbf{p}_i\|_2.
  $$
  
  This process ensures that each prototype is both representative and concise, making it easy to display the corresponding text as a human-readable explanation.
  **Notes:** I think this may be one incorrect procedure of this ProSeNet model, because of its reliance on the consistency of the underlying sequence encoder when processing reduced sequences. While models like BERT, with its masked language modeling pre-training, can naturally handle missing tokens, other sequence embedders—such as vanilla LSTMs or Bi-LSTMs trained on full, clean sequences—are more sensitive to token omissions. For example:
	- Original sequence: “I go to school in this weekend with my friends and I feel happy”
	- Optimal reduction: “I go school weekend with friends I happy” 
	=> The models can produce 2 embeddings that deviate significantly from the each other due to distributional drift. This indicates that the final reduction $\text{seq}_i$ may not be optimal. This can result in unpredictable and misleading reductions, potentially undermining the interpretability that ProSeNet aims to provide.

- **User-Guided Refinement:**  
  Recognizing that automatic prototype extraction may not always capture the nuanced patterns that domain experts expect, ProSeNet provides a mechanism for user interaction. Users can:
  - **Create** new prototypes when novel, significant patterns are identified.
  - **Revise** existing prototypes if the displayed sequence (or subsequence) does not fully reflect domain-relevant features.
  - **Delete** prototypes that are redundant or uninformative.
  
  Once a user commits to a refinement, the model is fine-tuned on the training data with the updated prototypes fixed. In this fine-tuning phase, rather than updating the latent prototype vectors via gradient descent, the sequence encoder $r$ is employed in every iteration to directly set:
  
  $$
  \mathbf{p}_i = r(\text{seq}_i).
  $$
  
  Moreover, the prototype projection step is skipped so that the user-specified prototypes remain unchanged. This interactive process allows the prototypes to incorporate external, domain-specific knowledge, thereby bolstering the interpretability of the model’s outputs.

---

### 3.2 Sample Visualization

- **Similarity Scoring:**  
  The input sequence is first encoded by the sequence encoder $r$ into a latent representation $e$. The similarity between $e$ and each prototype $\mathbf{p}_i$ is then computed via:
  
  $$
  a_i = \exp\left(-\|e - \mathbf{p}_i\|_2^2\right),
  $$
  
  which yields a similarity vector $\mathbf{a}$ with elements in the range $[0,1]$. A higher $a_i$ indicates a closer match between the input sequence and the prototype $\mathbf{p}_i$.

- **Weighted Explanation Construction:**  
  The final classification is obtained by aggregating the contributions of the prototypes through a weighted sum. The weights (often derived from the similarity scores) reflect the degree to which each prototype influences the prediction. For instance, given an input such as “pizza is good but service is extremely slow,” the model might explain its prediction as:
  
  $$
  0.69 \times \text{“good food but worst service” (Negative)} + 0.30 \times \text{“service is really slow” (Negative)}.
  $$
  
  This weighted combination clearly shows which prototype sequences contribute most to the final decision, thereby offering an interpretable rationale for the model’s output.

- **Interactive Feedback on Explanations:**  
  Beyond static visualization, users are encouraged to review the prototype-based explanations generated for each input. If an explanation is found to be misaligned with domain expectations—for example, if a prototype sequence includes extraneous or misleading text—users can intervene. Through the interactive refinement mechanism described in Section 3.1, they may adjust the set of prototypes. After such refinements, the model is re-fine-tuned (with prototypes held fixed) to better capture the domain-specific patterns, leading to more accurate and interpretable explanations.

---


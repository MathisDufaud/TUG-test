
from xml.parsers.expat import model
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, Bidirectional, LSTM, Concatenate, Lambda, Dense, TimeDistributed, Layer
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

# Phase labels and ordering
PHASE_LABELS = {
    '0:No test': 'No test',
    '1:Sit-to-stand': 'Sit-to-stand',
    '2:Turn1': 'Turn1',
    '3:Turn2+Stand-to-sit': 'Turn2+Stand-to-sit',
    '4:Walking1': 'Walking1',
    '5:Walking2': 'Walking2',
}

PHASE_ORDER = [
    'No test',
    'Sit-to-stand',
    'Walking1',
    'Turn1',
    'Walking2',
    'Turn2+Stand-to-sit',
]

# Create label to index mapping
PHASE_TO_IDX = {phase: idx for idx, phase in enumerate(PHASE_ORDER)}
IDX_TO_PHASE = {idx: phase for phase, idx in PHASE_TO_IDX.items()}
NUM_CLASSES = len(PHASE_ORDER)


class CRFLayer(Layer):
    """
    Conditional Random Field layer with transition constraints.
    Enforces that phases can only transition forward according to PHASE_ORDER.
    """
    
    def __init__(self, num_classes, **kwargs):
        super(CRFLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        
    def build(self, input_shape):
        # Transition matrix: [from_state, to_state]
        # Initialize with very negative values (impossible transitions)
        initial_transitions = np.full((self.num_classes, self.num_classes), -100, dtype=np.float32)
        # These are log-probabilities
        # -1e10 means probability ≈ 0 (exp(-1e10) ≈ 0)
        # During Viterbi, paths with -1e10 will never be chosen

        # Build allowed transitions based on PHASE_ORDER
        # 'No test' (idx 0) can transition to itself or 'Sit-to-stand' (idx 1)
        initial_transitions[0, 0] = 0.0  # No test -> No test
        initial_transitions[0, 1] = 0.0  # No test -> Sit-to-stand
        
        # Each phase can stay in itself or move to the next phase
        for i in range(1, self.num_classes):
            initial_transitions[i, i] = 0.0  # Stay in current phase
            if i < self.num_classes - 1:
                initial_transitions[i, i + 1] = 0.0  # Move to next phase
        #         To:    0      1      2      3      4      5
        # From: 0      0.0    0.0  -1e10  -1e10  -1e10  -1e10
        #       1    -1e10    0.0    0.0  -1e10  -1e10  -1e10
        #       2    -1e10  -1e10    0.0    0.0  -1e10  -1e10
        #       3    -1e10  -1e10  -1e10    0.0    0.0  -1e10
        #       4    -1e10  -1e10  -1e10  -1e10    0.0    0.0
        #       5      0.0  -1e10  -1e10  -1e10  -1e10    0.0
        
        # Last phase can transition back to 'No test'
        initial_transitions[self.num_classes - 1, 0] = 0.0
        
        #         This creates a learnable parameter that starts with our constraints
        # During training, valid transitions (0.0) can become positive or negative
        # Invalid transitions (-1e10) stay frozen (in practice, gradients won't make them valid)
        self.transitions = self.add_weight(
            name='transitions',
            shape=(self.num_classes, self.num_classes),
            initializer=tf.constant_initializer(initial_transitions),
            trainable=True
        )
        
        # Start and end transitions
        # Start transitions: How likely is each phase to be the FIRST phase?
        # End transitions: How likely is each phase to be the LAST phase?
        # Both are learned during training
        self.start_transitions = self.add_weight(
            name='start_transitions',
            shape=(self.num_classes,),
            initializer='zeros',
            trainable=True
        )
        
        self.end_transitions = self.add_weight(
            name='end_transitions',
            shape=(self.num_classes,),
            initializer='zeros',
            trainable=True
        )
        
        super(CRFLayer, self).build(input_shape)
    
    def call(self, inputs, mask=None, training=None):
        """
        During training: return emissions for loss computation
        During inference: return viterbi decoded sequence
        """
        if training:
            # Return emissions during training
            return inputs
        else:
            # Return decoded sequence during inference
            return self.viterbi_decode(inputs, mask)
    
    def viterbi_decode(self, emissions, mask=None):
        """
        Viterbi algorithm for finding most likely sequence.
        The Viterbi algorithm finds the highest-scoring valid path through all timesteps.
        i.e.: We have 3 timesteps
        At each timestep, the neural network gives emission scores for each phase
        We need to find the best sequence respecting transition constraints
        """
        batch_size = tf.shape(emissions)[0]
        seq_length = tf.shape(emissions)[1]
        
        # Initialize with start transitions
        score = emissions[:, 0, :] + self.start_transitions  # (batch, num_classes)
        
        # Store backpointers
        backpointers = []
        
        # Forward pass
        for i in range(1, emissions.shape[1]):
            # Expand dimensions for broadcasting
            score_expanded = tf.expand_dims(score, 2)  # (batch, num_classes, 1)
            emission_i = emissions[:, i, :]  # (batch, num_classes)
            
            # Calculate scores for all transitions
            next_score = score_expanded + self.transitions  # (batch, from_class, to_class)
            next_score = next_score + tf.expand_dims(emission_i, 1)  # (batch, from_class, to_class)
            
            # Find best previous state for each current state
            backpointer = tf.argmax(next_score, axis=1, output_type=tf.int32)  # (batch, num_classes)
            score = tf.reduce_max(next_score, axis=1)  # (batch, num_classes)
            # We need to remember which path we took to get the best score
            # After processing all timesteps, we'll trace backwards to reconstruct the path
            backpointers.append(backpointer)
        
        # Add end transitions
        score = score + self.end_transitions
        
        # Backward pass to get best path
        best_last_tag = tf.argmax(score, axis=1, output_type=tf.int32)  # (batch,)
        
        # Decode the best path
        best_tags = [best_last_tag]
        
        for backpointer in reversed(backpointers):
            best_last_tag = tf.gather_nd(
                backpointer,
                tf.stack([tf.range(batch_size), best_last_tag], axis=1)
            )
            best_tags.append(best_last_tag)
        
        # Reverse to get correct order and convert to float for consistency
        best_tags = tf.stack(list(reversed(best_tags)), axis=1)  # (batch, seq_length)
        
        return tf.cast(best_tags, tf.float32)
    
    def compute_loss(self, emissions, tags):
        """
        Compute CRF loss (negative log-likelihood).
        emissions: (batch_size, seq_length, num_classes)
        tags: (batch_size, seq_length) - ground truth labels

        CRF Loss = Log-Likelihood
        The loss is: loss = log(sum of all possible sequences) - log(correct sequence score)
        Intuition:

        gold_score: How good is the TRUE sequence according to our model?
        norm_score: How good are ALL POSSIBLE sequences combined?
        We want: gold_score to be high, other sequences to be low
        Minimizing this loss makes the correct sequence have higher probability
        """

        # Calculate score for the gold sequence
        gold_score = self.score_sentence(emissions, tags)
        
        # Calculate normalization (all possible sequences)
        norm_score = self.log_norm(emissions)
        
        # Loss is negative log-likelihood
        loss = norm_score - gold_score
        
        return tf.reduce_mean(loss)
    
    def score_sentence(self, emissions, tags):
        """
        Score of a given tag sequence.
        1: Score for starting in the gold first phase
        Get start transition for true first tag
        Add emission score for true first tag

        2: For each consecutive pair of tags: tags[i-1] → tags[i]
        Get transition score: transitions[tags[i-1], tags[i]]
        Get emission score: emissions[i, tags[i]]
        Add both to running score

        3: Add end transition
        
        """
        batch_size = tf.shape(emissions)[0]
        seq_length = tf.shape(emissions)[1]
        tags = tf.cast(tags, tf.int32)
        
        # Start transitions
        score = tf.gather(self.start_transitions, tags[:, 0])
        score += tf.gather_nd(emissions[:, 0, :], 
                             tf.stack([tf.range(batch_size), tags[:, 0]], axis=1))
        
        # Transitions and emissions
        for i in range(1, emissions.shape[1]):
            indices = tf.stack([tags[:, i-1], tags[:, i]], axis=1)
            transition_score = tf.gather_nd(self.transitions, indices)
            
            emission_indices = tf.stack([tf.range(batch_size), tags[:, i]], axis=1)
            emission_score = tf.gather_nd(emissions[:, i, :], emission_indices)
            
            score += transition_score + emission_score
        
        # End transitions
        score += tf.gather(self.end_transitions, tags[:, -1])
        
        return score
    
    def log_norm(self, emissions):
        """Log-sum-exp of all possible sequences (partition function)."""
        seq_length = tf.shape(emissions)[1]
        
        # Initialize with start transitions
        score = emissions[:, 0, :] + self.start_transitions
        
        # Forward algorithm
        for i in range(1, emissions.shape[1]):
            score_expanded = tf.expand_dims(score, 2)
            emission_i = tf.expand_dims(emissions[:, i, :], 1)
            
            next_score = score_expanded + self.transitions + emission_i
            score = tf.reduce_logsumexp(next_score, axis=1)
        
        # Add end transitions
        score = score + self.end_transitions
        
        return tf.reduce_logsumexp(score, axis=1)
    
    def get_config(self):
        config = super(CRFLayer, self).get_config()
        config.update({'num_classes': self.num_classes})
        return config



class CRFModel(Model):
    """
    Custom Model class that handles CRF loss computation.
    """
    def __init__(self, inputs, outputs, crf_layer, emissions_output, **kwargs):
        super(CRFModel, self).__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.crf_layer = crf_layer
        self.emissions_output = emissions_output

    def train_step(self, data): 
        x, y = data
        # Squeeze y if it has extra dimension: (batch, seq, 1) -> (batch, seq)
        if len(y.shape) == 3 and y.shape[-1] == 1:
            y = tf.squeeze(y, axis=-1)
        
        with tf.GradientTape() as tape:
            # Get emissions (forward pass with training=True)
            emissions = self.emissions_output(x, training=True)
            
            # Compute CRF loss
            loss = self.crf_layer.compute_loss(emissions, y)
            
            # Add regularization losses if any
            if self.losses:
                loss += tf.add_n(self.losses)
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Get predictions for metrics (viterbi decode)
        y_pred = self.crf_layer.viterbi_decode(emissions)
        
        # Update metrics
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
      
    def test_step(self, data): 
        
        x, y = data        
        # Squeeze y if it has extra dimension: (batch, seq, 1) -> (batch, seq)
        if len(y.shape) == 3 and y.shape[-1] == 1:
            y = tf.squeeze(y, axis=-1)
        
        # Get emissions
        emissions = self.emissions_output(x, training=False)

        # Compute CRF loss
        loss = self.crf_layer.compute_loss(emissions, y)
        
        # Get predictions (viterbi decode)
        y_pred = self.crf_layer.viterbi_decode(emissions)
        
        # Update metrics
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    def call(self, inputs, training=None):
        """
        Forward pass.
        During training: returns emissions
        During inference: returns viterbi decoded sequence
        """
        emissions = self.emissions_output(inputs, training=training)
        if training:
            return emissions
        else:
            return self.crf_layer.viterbi_decode(emissions)


def build_crf_model(window_size=60, n_features=9, output_steps=15):
    """
    Build the CRF-enhanced model for phase prediction.
    Use this exactly like your original model!
    """
    inp = Input(shape=(window_size, n_features))
    
    # Multi-scale convolutions
    c1 = Conv1D(64, 3, padding='same', activation='relu')(inp)
    c1 = BatchNormalization()(c1)
    
    c2 = Conv1D(64, 5, padding='same', activation='relu')(inp)
    c2 = BatchNormalization()(c2)
    
    c3 = Conv1D(64, 7, padding='same', activation='relu')(inp)
    c3 = BatchNormalization()(c3)
    
    x = Concatenate()([c1, c2, c3])  # (batch, 60, 192)
    x = Dropout(0.2)(x)
    
    # Temporal modeling
    x = Bidirectional(LSTM(64, return_sequences=True))(x)  # (batch, 60, 128)
    x = Dropout(0.3)(x)
    
    # Keep only the last output_steps timesteps
    x = Lambda(lambda t: t[:, -output_steps:, :])(x)  # (batch, 15, 128)
    
    # Emission scores (unnormalized log probabilities)
        # Why activation='linear' instead of softmax?
        # In your original model:

        # softmax converts scores to probabilities: [0.1, 0.7, 0.2, ...]
        # Each timestep is independent

        # In CRF:

        # We need unnormalized scores (emissions)
        # CRF layer handles normalization globally across the whole sequence
        # This allows transitions to influence the final probabilities

    emissions = TimeDistributed(
        Dense(NUM_CLASSES, activation=None,
            kernel_initializer="glorot_uniform")
    )(x)

    emissions = tf.keras.layers.LayerNormalization(axis=-1)(emissions)

    # Create emissions model (for computing emissions)
    emissions_model = Model(inp, emissions)
    
    # CRF layer for constrained decoding
    crf_layer = CRFLayer(NUM_CLASSES)
    predictions = crf_layer(emissions)  # (batch, 15)
    
    # Create the full model
    model = CRFModel(
        inputs=inp,
        outputs=predictions,
        crf_layer=crf_layer,
        emissions_output=emissions_model
    )

    return model

def define_results_crf(all_tests, method):
    all_results = {}
    for test in all_tests:
        if test.results != None and test.results[method] != None and test.results[method] != '':
            all_results[str(test.user_id) + '_' + str(test.session_id) + '_' + test.context[0]] = test.results[method]

    return all_results

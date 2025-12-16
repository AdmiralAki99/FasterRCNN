import math
import tensorflow as tf


# Creatig a custom learning rate scheduler

class CustomStepBasedScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_learning_rate, total_steps, warmup_steps, hold_steps, minimum_learning_rate):
        super(CustomStepBasedScheduler, self).__init__()
        self.base_learning_rate = tf.cast(base_learning_rate, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.hold_steps = tf.cast(hold_steps, tf.float32)
        self.minimum_learning_rate = tf.cast(minimum_learning_rate, tf.float32)
        
    def __init__(self, base_learning_rate, steps_per_epoch, total_epochs, warmup_steps, hold_steps, minimum_learning_rate):
        super(CustomStepBasedScheduler, self).__init__()
        self.base_learning_rate = tf.cast(base_learning_rate, tf.float32)
        self.steps_per_epoch = tf.cast(steps_per_epoch, tf.float32)
        self.total_steps = tf.cast(steps_per_epoch * total_epochs, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.hold_steps = tf.cast(hold_steps, tf.float32)
        self.minimum_learning_rate = tf.cast(minimum_learning_rate, tf.float32)
        
    # Define the learning rate schedule
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # Linear warmup
        # Guarding for when warmup_steps is 0
        warmup_learning_rate = self.base_learning_rate * (step / tf.maximum(1.0, self.warmup_steps))
        
        # Hold phase
        hold_learning_rate = self.base_learning_rate
        
        # Cosine decay phase
        decay_steps = tf.maximum(1.0, self.total_steps - self.warmup_steps - self.hold_steps)
        t = tf.clip_by_value((step - self.warmup_steps - self.hold_steps) / decay_steps, 0.0, 1.0)
        cosine_learning_rate = self.minimum_learning_rate + (0.5 * (self.base_learning_rate - self.minimum_learning_rate) * (1.0 + tf.cos(math.pi * t)))
        
        # Conditional logic to select the correct learning rate phase
        learning_rate = tf.where(step < self.warmup_steps, warmup_learning_rate, tf.where(step < self.warmup_steps + self.hold_steps, hold_learning_rate, cosine_learning_rate))
        
        return learning_rate
        
        
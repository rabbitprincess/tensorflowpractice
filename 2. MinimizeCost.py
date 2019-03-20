import tensorflow as tf
import matplotlib.pyplot as plt #matplotlib 사용

X = [1,2,3,4,5]
Y = [3,6,9,12,15]

W = tf.placeholder(tf.float32) # float32를 받는 placeholder
hypothesis = X*W
cost = tf.reduce_mean(tf.square(hypothesis-Y)) #cost 식 그대로 사용
sess = tf.Session()
sess.run(tf.global_variables_initializer())
W_val = [] #weight의 값을 저장할 list
cost_val = [] # cost의 값을 저장할 list
for i in range(-30,30):
    feed_W = i
    curr_cost, curr_W = sess.run([cost,W], feed_dict = {W:feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)
    
plt.plot(W_val, cost_val)
plt.show()# W가 3의 값을 가질때 cost가 최소가 된다!
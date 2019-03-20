import tensorflow as tf


#hello = tf.constant('hello, Tensorflow!')
#sess = tf.Session()
#print(sess.run(hello))

sess = tf.Session() # Session을 시작하는데 필요한 변수
# x and y data
x_train = [1,2,3] #x가 ㄴ1,2,3일때
y_train = [1,2,3] #y가 1,2,3이 나오는 값을 예측한다.
# Weight and bias
W = tf.Variable(tf.random_normal([1]), name='weight') #weight
b = tf.Variable(tf.random_normal([1]), name='bias') #bias

hypothesis = x_train*W + b # hypothesis
cost = tf.reduce_mean(tf.square(hypothesis - y_train)) #cost

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01) # 경사 하강법
train = optimizer.minimize(cost) # optimizer(cost 최적화)


sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step<20 : 
        print(step, sess.run(cost),sess.run(W),sess.run(b))
    elif step%50==0:
        print(step, sess.run(cost),sess.run(W),sess.run(b))
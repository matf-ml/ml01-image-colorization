n = 10000 # 
X = np.arange(n)[:,None]
print(X.shape)
y = np.random.randn(n)[:,None]*500 + X
#y = y.reshape(n,1)

#plt.plot(X[:,1], y)
yval = y
X = np.hstack((X, yval))
# labels: 1 if above line, 0  below
print(y.shape, X[:,0].shape)
y = y > X[:,0][:,None]
print(y.shape, X.shape)
y = y.reshape(n,1)
y = np.hstack((y, ~y )).astype('uint8')
print(y.shape)
colors = ['red' if i  else 'green' for i in y[:,0]]
plt.scatter(X[:,0], X[:,1], color=colors)
plt.show()
numRed = len([x for x in colors if x=='red'])
print('red:',numRed, 'green:',len(colors) - numRed)
print(X.shape, y.shape)
print(y[-10:])
print(X[-10:])

def createTestModel(layers):
    model = Sequential()
    #print(X.shape[1])
    model.add(Dense(layers[0], input_dim=X.shape[1], kernel_regularizer=l2(0.001)) ) 
    model.add(Activation('relu'))
    for l in layers[1:]:
        model.add(Dense(l, kernel_regularizer=l2(0.001)))
        model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                 metrics=['accuracy'])
    return model

model = createTestModel([3,2])
model.fit(X, y[:,0], epochs=7)
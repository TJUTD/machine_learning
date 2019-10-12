'''
CS6375 Machine Learning
Decison Tree  
'''

import sys
import math
import pandas as pd
import numpy as np
import argparse



class treeNode:
	def __init__(self, attr):
		self.left = None;
		self.leftClassCount = 0; # for pruning
		self.depth = -1; 
		self.right = None;
		self.rightClassCount = 0; # for pruning
		self.misCount = 0; # for reduced-error pruning
		self.attribute = attr;
		self.decision = None;
		self.id = None;

class dTree:
	root = None # root of tree
	df = None # training set data frame
	impurity= None # impurity  heuristic
	height = -1  # height of tree
	
	def __init__(self, df, impurity):
		self.df = df 
		self.impurity = impurity

	def train(self):	
		featSet = set(range(0, self.df.shape[1]-1)) 
		if self.impurity == 'en':
			# print('build_tree_info_gain')
			nextAttr = best_attr_info_gain(self.df, featSet)
			self.root = treeNode(nextAttr)
			self.build_tree_info_gain(self.root, self.df, featSet)
		else :
			# print('build_tree_var_impure')
			nextAttr = best_attr_var_impure(self.df, featSet)
			self.root = treeNode(nextAttr)
			self.build_tree_var_impure(self.root, self.df, featSet)
			
		
	'''
	train by information gain heuristic
	inputes:
		- root node
		- data frame
		- feature set
	'''
	def build_tree_info_gain(self, root, df, featSet):
		# print('build_tree_info_gain')
		if(root.attribute == -1):
			# print("no more")
			return
		featSet.remove(root.attribute)
		nrow = df.shape[0]
		
		# left branch (0)
		dfLeft = df[df.iloc[:,root.attribute]==0]
		nrowLeft = dfLeft.shape[0] 
		# print(nrowLeft)
		nrowRight = nrow-nrowLeft
		posCount = sum(df.iloc[:,-1])
		root.leftClassCount = nrow - posCount
		leftAttribute = best_attr_info_gain(dfLeft, featSet)
		root.left = treeNode(leftAttribute)
						
		#no further split, check remain test case, choose majority
		if leftAttribute == -1:
			posCountLeft = sum(dfLeft.iloc[:,-1])
			root.left.decision = 1 if posCountLeft > nrowLeft - posCountLeft else 0
		
		self.build_tree_info_gain(root.left, dfLeft, featSet.copy())
				
		# right branch (1)
		dfRight = df[df.iloc[:,root.attribute]==1]
		root.rightClassCount = posCount
		rightAttribute = best_attr_info_gain(dfRight, featSet)
		root.right = treeNode(rightAttribute)
		
		#no further split, check remain test case, choose majority
		if rightAttribute == -1:
			posCountRight = sum(dfRight.iloc[:,-1])
			root.right.decision = 1 if posCountRight > nrowRight - posCountRight else 0
			
		self.build_tree_info_gain(root.right, dfRight, featSet.copy())
		
		
	'''
	train by variance impurity heuristic
	inputes:
		- root node
		- data frame
		- feature set
	'''
	def build_tree_var_impure(self, root, df, featSet):
		# print('build_tree_var_impure')
		if(root.attribute == -1):
			# print("no more")
			return
		featSet.remove(root.attribute)
		nrow = df.shape[0]
		
		# left branch (0)
		dfLeft = df[df.iloc[:,root.attribute]==0]
		nrowLeft = dfLeft.shape[0] 
		nrowRight = nrow-nrowLeft
		posCount = sum(df.iloc[:,-1])
		root.leftClassCount = nrow - posCount
		leftAttribute = best_attr_var_impure(dfLeft, featSet)
		root.left = treeNode(leftAttribute)
						
		#no further split, check remain test case, choose majority
		if leftAttribute == -1:
			posCountLeft = sum(dfLeft.iloc[:,-1])
			root.left.decision = 1 if posCountLeft > nrowLeft - posCountLeft else 0
		
		self.build_tree_var_impure(root.left, dfLeft, featSet.copy())
				
		# right branch (1)
		dfRight = df[df.iloc[:,root.attribute]==1]
		root.rightClassCount = posCount
		rightAttribute = best_attr_var_impure(dfRight, featSet)		
		root.right = treeNode(rightAttribute)
		
		#no further split, check remain test case, choose majority
		if rightAttribute == -1:
			posCountRight = sum(dfRight.iloc[:,-1])
			root.right.decision = 1 if posCountRight > nrowRight - posCountRight else 0
		self.build_tree_var_impure(root.right, dfRight, featSet.copy())

	
	
	'''
	predict the class of given record
	inputes:
		- root node
		- record row
	output:
		- class prediction
	'''
	def predict(self, root, row):
		if root!=None:
			curr = root
			while(curr.attribute!=-1):
				attr = curr.attribute
				
				# need find a way to avoid data type implicit convert
				if row[attr] == 0:
					curr = curr.left
				else:
					curr = curr.right
			return curr.decision if curr.decision!='' else None
		else:
			print("need train first")

	
	'''
	prediction accuracy on given dataset
	inputes:
		- dataset
	output:
		- classification accuracy
	'''
	def computeAccuracy(self, df):
		nrow = df.shape[0]
		matchCount = 0.0
		if self.root!= None:
			for row in range(0, df.shape[0]):
				currRow = df.iloc[row]
				predictResult = self.predict(self.root, currRow)
				if predictResult == currRow.iloc[-1]:
					matchCount += 1
			return matchCount/nrow
		else:
			print("need train first")
			
			
	'''pruning the tree, Reduce-Error pruning
	inputs:
		- validation set 
	'''
	def pruning_RE(self, validateDF):
		# print('pruning_RE')
		if self.root != None:
			validArruracy = self.computeAccuracy_RE(validateDF)
			
			self.depth_RE(self.root, 0)
			# print('height ', self.height - 1)
			depth = self.height - 1
			while depth > 0 :
				levelSet = self.innerNode(depth)
				for node in levelSet:
					self.prune(node)
				depth = depth - 1
		else:
			print("need train first")
			
			
	'''pruning the tree, Depth-Based pruning
	inputs:
		- validation set 
	'''
	def pruning_DB(self, validateDF):
		# print('pruning_DB')
		if self.root != None:
			validArruracy = self.computeAccuracy(validateDF)
			self.depth(self.root, 0)
			
			dmax = (100,50,20,15,10,5) 
			
			prunedTree = None
			
			for dm in dmax :
				# print(dm)
				# deep copy to generate new root
				prunedTree = self.dmaxDT(self.root, dm)
				# validation accuracy after pruning
				currAcc = self.currentAccuracy(prunedTree, validateDF)
				# print('d_max ', dm, ' accuracy', currAcc)
				if(currAcc > validArruracy):
					# print('dmax ',dm)						
					validArruracy = currAcc
					self.root = prunedTree
								
		else:
			print("need train first")	
			
	
	'''
	mark the depth of each tree node
	inputes:
		- root node
		- depth of root
	'''
	def depth(self, root, dep):
		if root == None :
			return
		root.depth = dep
		self.depth(root.left, dep+1)
		self.depth(root.right, dep+1)
		
	
	'''
	copy decision tree with height <= dep
	inputes:
		- root node
		- given height
	output:
		- root of pruned tree 
	'''
	def dmaxDT(self, root, dep) :
		if root == None:
			return None
		if root.depth == dep:
			newNode = treeNode(-1)
			newNode.depth = root.depth
			if root.attribute == -1 :
				newNode.decision = root.decision
			else :
				newNode.decision = 1 if root.rightClassCount > root.leftClassCount else 0
			return newNode
		else:
			newNode = treeNode(root.attribute)
			newNode.decision = root.decision
			newNode.leftClassCount = root.leftClassCount
			newNode.rightClassCount = root.rightClassCount
			newNode.depth = root.depth
			newNode.left = self.dmaxDT(root.left, dep)
			newNode.right = self.dmaxDT(root.right, dep)
			return newNode
		

	'''
	validate of rows of data  based on new decision tree afte pruning
	inputes:
		- root node
		- date frame
	ouptut:
		- accuracy of pruned tree
	'''
	def currentAccuracy(self, root, df):
		nrow = df.shape[0]
		matchCount = 0.0
		if root!= None:
			for row in range(0, df.shape[0]):
				currRow = df.iloc[row]
				pred = self.predict(root, currRow)
				if pred == currRow.iloc[-1]:
					matchCount += 1
			return matchCount/nrow
		else:
			print("need a pruned tree")
	
	'''
	predict the class of given record for reduced-error pruning
	inputes:
		- root node
		- record row
	output:
		- class prediction
	'''
	def predict_RE(self, root, row):
		if root!=None:
			curr = root
			while(curr.attribute!=-1):
				attr = curr.attribute
				currdecision = 1 if curr.rightClassCount > curr.leftClassCount else 0
				if currdecision != row.iloc[-1] :
					curr.misCount += 1 
				
				# need find a way to avoid data type implicit convert
				if row[attr] == 0:
					curr = curr.left
				else:
					curr = curr.right
			if curr.decision != '' :
				if curr.decision != row.iloc[-1] :
					curr.misCount += 1
				return curr.decision
			else :
				return None
		else:
			print("need train first")

	'''
	prediction accuracy on given dataset for reduced-error pruning
	inputes:
		- dataset
	output:
		- classification accuracy
	'''
	def computeAccuracy_RE(self, df):
		nrow = df.shape[0]
		matchCount = 0.0
		if self.root!= None:
			for row in range(0, df.shape[0]):
				currRow = df.iloc[row]
				pred = self.predict_RE(self.root, currRow)
				if pred == currRow.iloc[-1]:
					matchCount += 1
			return matchCount/nrow
		else:
			print("need train first")
	
	'''
	mark the depth of each tree node and record the height of tree for reduced-error pruning
	inputes:
		- root node
		- depth of root
	'''
	def depth_RE(self, root, dep):
		if root == None :
			return
		if dep > self.height :
			self.height = dep 
		root.depth = dep
		self.depth_RE(root.left, dep+1)
		self.depth_RE(root.right, dep+1)
	
	'''
	find inner nodes of tree at given depth
	inputes:
		- root node
		- depth of candidate nodes
	output:
		- set of inner nodes
	'''
	def innerNode(self, depth):
		levelset = set()
		queue= []
		queue.append(self.root)
		while(len(queue)>0):
			tmp = queue.pop(0)
			if tmp.attribute != -1 and tmp.depth == depth:
				levelset.add(tmp)
			if tmp.left != None:
				queue.append(tmp.left)
			if tmp.right != None:
				queue.append(tmp.right)
		return levelset
		
	'''
	prune the subtree of given node having less error than its subtree
	inputes:
		- root node
	'''
	def prune(self, node) :
		if node == None :
			return
		errTree = self.error_of_subtree(node)
		errLeaf = node.misCount
		if errLeaf <= errTree : 
			node.attribute = -1
			node.decision = 1 if node.rightClassCount > node.leftClassCount else 0
			node.left = None
			node.right = None
	
	'''
	compute the error of subtree 
	inputes:
		- root node
	'''
	def error_of_subtree(self, node) :
		mis = 0
		queue= []
		queue.append(node)
		while(len(queue)>0):
			tmp = queue.pop(0)
			if tmp.attribute == -1:
				mis += tmp.misCount
			if tmp.left != None:
				queue.append(tmp.left)
			if tmp.right != None:
				queue.append(tmp.right)
		return mis
	


# utility function

'''
select the best split feature using information gain heuristic
inputs:
	- dataframe
	- feature set
return: the index of the best split feature
'''
def best_attr_info_gain(df, featSet):
	maxVal = - 0.1
	nextAttribute = -1
	if df.shape[0] == 0:
		return nextAttribute
	
	S0 =  entropy(df.iloc[:,-1])
	'''
	under below condition, return -1
	- pure
	- no more attribute to split 
	- all data attribute in df same 
	'''
	if S0 == 0 or df.shape[0]<=1:
		return nextAttribute
	if  df.iloc[:, list(featSet)].drop_duplicates().shape[0] ==1:
		return nextAttribute
	for feat in featSet:
		currIG = info_gain(df, S0, feat)
		if currIG > maxVal:
			maxVal = currIG
			nextAttribute = feat
	return nextAttribute 

'''
calculate entropy of given binary column
input:
	- column
return: entropy 
'''
def entropy(col):
	total = len(col)
	S = 0
	prob = float(sum(col))/total
	if prob != 0 and prob != 1:
		S = - prob * math.log(prob,2) - (1-prob) * math.log(1-prob,2)
	return S

'''
compute information gain of given entropy and possible attribute
inputs:
	- dataframe
	- entropy of Given, S_curr
	- next attribute, feat
return: information gain
'''
def info_gain(df, S_curr, feat):
	total = df.shape[0];
	left =  df.loc[df.iloc[:,feat]==0].iloc[:,-1]
	right =  df.loc[df.iloc[:,feat]==1].iloc[:,-1]
	frac = float(len(left)) / total
	infoGain = S_curr
	if frac != 0:
		infoGain -= frac * entropy(left)
	if frac != 1:
		infoGain -= (1-frac) * entropy(right)
	return infoGain
	
'''
select the best split feature using variance impurity heuristic
inputs:
	- dataframe
	- feature set
return: the index of the best split feature
'''
def best_attr_var_impure(df, featSet):
	maxVal = - 0.1
	nextAttribute = -1
	K = df.shape[0]
	if K == 0:
		return nextAttribute
		
	K1 = sum(df.iloc[:,-1])
	K0 = K - K1
	VI = 1.0 * K0 * K1 / K / K
	'''
	under below condition, return ''
	- pure
	- no more attribute to split 
	- all data attribute in df same 
	'''
	if VI == 0 or K <= 1:
		return nextAttribute
	if df.iloc[:, list(featSet)].drop_duplicates().shape[0] ==1:
		return nextAttribute
		
	for feat in featSet:
		currVG = VI_gain(df, VI, feat)
		if currVG > maxVal:
			maxVal = currVG
			nextAttribute = feat
	return nextAttribute
	
'''
compute gain of given variance impurity and possible attribute
inputs:
	- dataframe
	- variance impurity, VI_curr
	- next attribute, feat
return: variance impurity gain
'''
def VI_gain(df, VI_curr, feat):
	total = df.shape[0];
	left =  df.loc[df.iloc[:,feat]==0].iloc[:,-1]
	right =  df.loc[df.iloc[:,feat]==1].iloc[:,-1]
	frac = float(len(left)) / total
	varGain = VI_curr
	if frac != 0:
		K_L = len(left)
		K1_L = sum(left)
		K0_L = K_L - K1_L
		varGain -= frac * K0_L * K1_L / K_L / K_L
	if frac != 1:	
		K_R = len(right)
		K1_R = sum(right)
		K0_R = K_R - K1_R
		varGain -= frac * K0_R * K1_R / K_R / K_R
	return varGain	
	
	
'''
decision tree learner
inputs:
	- number of clause of the dataset
	- number of '+/-' pairs of the date sset
	- impurity heuristic, 'en' or 'va'
	- pruning method, 'no', 're' or 'db' 
	- training set
	- test set
 	- validation set
return: variance impurity gain
'''
def dtree_learner(clause, dim, impurity, prune, trainDF, testDF, validDF):
	mydTree = dTree(trainDF, impurity)
	mydTree.train()
	# print('training...')
	if prune == 'no' :
		if impurity == 'en' :
			print(' Accuracy of Naive Decision Tree with Entropy as the impurity heuristic for data sets c = ', clause, ' d = ', dim, ': ', mydTree.computeAccuracy(testDF))
		else:
			print(' Accuracy of Naive Decision Tree with Variance as the impurity heuristic for data sets c = ', clause, ' d = ', dim, ': ', mydTree.computeAccuracy(testDF))
	else: 
		if prune == 're':
			# print(' Accuracy of before pruning: ', mydTree.computeAccuracy(testDF))
			mydTree.pruning_RE(validDF)
			# print('pruning...')
			if impurity == 'en' :
				print(' Accuracy of Decision Tree with Entropy as the impurity heuristic and Reduced-Error pruning for data sets c = ', clause, ' d = ', dim, ': ', mydTree.computeAccuracy(testDF))
			else :
				print(' Accuracy of Decision Tree with Variance as the impurity heuristic and Reduced-Error pruning for data sets c = ', clause, ' d = ', dim, ': ', mydTree.computeAccuracy(testDF))
		else :
			# print(' Accuracy of before pruning: ', mydTree.computeAccuracy(testDF))
			mydTree.pruning_DB(validDF)
			# print('pruning...')
			if impurity == 'en' :
				print(' Accuracy of Decision Tree with Entropy as the impurity heuristic and Depth-Based pruning for data sets c = ', clause, ' d = ', dim, ': ', mydTree.computeAccuracy(testDF))
			else:
				print(' Accuracy of Decision Tree with Variance as the impurity heuristic and Depth-Based pruning for data sets c = ', clause, ' d = ', dim, ': ', mydTree.computeAccuracy(testDF))
				
 
def main(args):
	train_file = args.data_dir +'/train_c' + str(args.clause) + '_d' + str(args.dim) + '.csv'
	valid_file = args.data_dir +'/valid_c' + str(args.clause) + '_d' + str(args.dim) + '.csv'
	test_file = args.data_dir +'/test_c' + str(args.clause) + '_d' + str(args.dim) + '.csv'
	
	trainDataFrame = pd.read_csv(train_file,header = None)
	trainDataFrame.columns = ['X' + str(i) for i in range(1,trainDataFrame.shape[1])] + ['Y']
	
	testDataFrame = pd.read_csv(test_file,header = None)
	testDataFrame.columns = ['X' + str(i) for i in range(1,testDataFrame.shape[1])] + ['Y']
	
	validateDataFrame = pd.read_csv(valid_file,header = None)
	validateDataFrame.columns = ['X' + str(i) for i in range(1,validateDataFrame.shape[1])] + ['Y']
	
		
	if args.method == 'rf' :
		from sklearn.ensemble import RandomForestClassifier
	
		# Create a Gaussian Classifier
		clf=RandomForestClassifier()
		
		ncol = trainDataFrame.shape[1] 
		x_train = trainDataFrame.iloc[:,list(range(0,ncol-2))] 
		y_train = trainDataFrame.iloc[:,ncol-1] 
		x_test = testDataFrame.iloc[:,list(range(0,ncol-2))] 
		y_test = testDataFrame.iloc[:,ncol-1] 
		
		# Train the model using the training sets y_pred=clf.predict(X_test)
		clf.fit(x_train, y_train)

		# prediction on test set
		y_pred=clf.predict(x_test)

		# Import scikit-learn metrics module for accuracy calculation
		from sklearn import metrics
		print(' Accuracy of Random Forest for data sets c = ', args.clause, ' d = ', args.dim, ': ', metrics.accuracy_score(y_test, y_pred))
	
	else:
		dtree_learner(args.clause, args.dim, args.impurity, args.prune, trainDataFrame, testDataFrame, validateDataFrame)
		

	
if __name__ == '__main__' :
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type = str, default = './all_data', help = 'path to the data sets')
	parser.add_argument('--clause', type = int, default = 300, help = 'number of clauses')
	parser.add_argument('--dim', type=int, default = 100, help='number of positive or negative examples ')
	parser.add_argument('--method', choices=['dt', 'rf'], default="dt", help="decision tree or random forest")
	parser.add_argument('--impurity', choices=['en', 'va'], default="en", help='impurity heuristic: entropy or variance')
	parser.add_argument('--prune', choices=['no', 're', 'db'], default="no", help='pruning method: none, reduced-error, depth-based')
	
	args = parser.parse_args()
	main(args)
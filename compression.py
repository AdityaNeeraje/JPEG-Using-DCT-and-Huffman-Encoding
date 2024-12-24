from PIL import Image
import numpy as np
from scipy.fft import dct, idct
from queue import PriorityQueue

image=Image.open('coil-20-unproc/obj1__0.png')
image_array=np.array(image)

image_numbers=image_array.flatten()

array=image_array

def type_2_dct_without_orthonormalization(array):
    length=8
    cosine_matrix=2*np.cos(np.pi/(2*length)*np.outer(np.arange(length), 2*np.arange(length)+1))
    new_array=np.zeros_like(array, dtype=np.float64)
    for i in range(0, len(array), length):
        if i+length>len(array):
            l=len(array)-i
            new_cosine_matrix=2*np.cos(np.pi/(2*l)*np.outer(np.arange(l), 2*np.arange(l)+1))
            new_array[i:i+l]=np.inner(array[i:i+l], new_cosine_matrix)
            continue
        new_array[i:i+length]=np.inner(array[i:i+length], cosine_matrix)
    return new_array

def type_2_dct_with_orthonormalization(array):
    length=8
    cosine_matrix=2*np.cos(np.pi/(2*length)*np.outer(np.arange(length), 2*np.arange(length)+1))
    cosine_matrix[0]/=np.sqrt(2)
    cosine_matrix/=np.sqrt(2*length)
    new_array=np.zeros_like(array, dtype=np.float64)
    for i in range(0, len(array), length):
        if i+length>len(array):
            l=len(array)-i
            new_cosine_matrix=2*np.cos(np.pi/(2*l)*np.outer(np.arange(l), 2*np.arange(l)+1))
            new_cosine_matrix[0]/=np.sqrt(2)
            new_cosine_matrix/=np.sqrt(2*l)
            new_array[i:i+l]=np.inner(array[i:i+l], new_cosine_matrix)
            continue
        new_array[i:i+length]=np.inner(array[i:i+length], cosine_matrix)
    return new_array

def two_dimensional_dct(array):
    dct_image_array=np.array([type_2_dct_with_orthonormalization(array[i]) for i in range(len(array))])
    return np.array([type_2_dct_with_orthonormalization(dct_image_array.T[i]) for i in range(len(dct_image_array.T))])
                               
def standard_dct(array, type=2, ortho=True):
    length=8
    if not ortho:
        norm='backward'
    else:
        norm='ortho'
    new_array=np.zeros_like(array, dtype=np.float64)
    for i in range(0, len(array), length):
        if i+length>len(array):
            l=len(array)-i
            new_array[i:i+l]=dct(array[i:i+l], norm=norm)
            continue
        new_array[i:i+length]=dct(array[i:i+length], norm=norm)
    return new_array

### Testing 2D DCT 
# standard_dct_image_array=np.array([standard_dct(array[i]) for i in range(len(array))])
# standard_dct_image_array_2=np.array([standard_dct(standard_dct_image_array.T[i]) for i in range(len(standard_dct_image_array.T))])
                                
# print(np.linalg.norm(two_dimensional_dct(image_array)-standard_dct_image_array_2))
result=(np.round(two_dimensional_dct(image_array)/16)).flatten()
# Note that 0 is actually slightly better than 1 as default, since 1 -> 3 bits, 0 -> 1 bit later on
rle_result=[result[0], 0]
for value in result[1:]:
    if value==rle_result[-2]:
        rle_result[-1]+=1
    else:
        rle_result.append(value)
        rle_result.append(0)
rle_result=np.array(rle_result)

class TrieNode:
    def __init__(self, value=1e10, frequency=0):
        self.frequency=frequency
        self.left=None
        self.right=None
        self.is_end_of_word=value

    def __lt__(self, other):
        return self.frequency<other.frequency   
    
class HuffmanTrie:
    def __init__(self, int_list):
        self.root=None
        self.character_encoding={}
        result=self.encode(int_list)
        print(len(result))
        assert (self.decode(result)==int_list).all()
    
    def insert(self, string, value):
        node=self.root
        for char in string:
            if char=='0':
                if not node.left:
                    node.left=TrieNode()
                node=node.left
            else:
                if not node.right:
                    node.right=TrieNode()
                node=node.right
        node.is_end_of_word=value
    
    def encode(self, int_list):
        frequency_dict={}
        for value in int_list:
            if value not in frequency_dict:
                frequency_dict[value]=1
            else:
                frequency_dict[value]+=1
        pq=PriorityQueue()
        for key in frequency_dict:
            pq.put(TrieNode(value=key, frequency=frequency_dict[key]))
        while pq.qsize()>1:
            left=pq.get()
            right=pq.get()
            new_node=TrieNode(frequency=left.frequency+right.frequency)
            new_node.left=left
            new_node.right=right
            pq.put(new_node) 
        self.root=pq.get()     
        self.dfs_to_fill_character_encoding()
        result=""
        for value in int_list:
            result+=self.character_encoding[value]
        return result

    def dfs_to_fill_character_encoding(self):
        def dfs(node, string):
            if not node:
                return
            if node.is_end_of_word!=1e10:
                self.character_encoding[node.is_end_of_word]=string
            dfs(node.left, string+'0')
            dfs(node.right, string+'1')
        dfs(self.root, '')
    
    def decode(self, string):
        result=[]
        node=self.root
        for char in string:
            if char=='0':
                node=node.left
            else:
                node=node.right
            if node.is_end_of_word!=1e10:
                result.append(node.is_end_of_word)
                node=self.root
        return result
ht=HuffmanTrie(rle_result)
# ht.encode()

# TODO -> Implement Inverse DCT, figure out the quantization matrix and plot the error vs byte storage. Try to implement Lempel-Ziv for accurate compression.
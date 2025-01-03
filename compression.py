from PIL import Image
import numpy as np
from scipy.fft import dct, idct
from queue import PriorityQueue
import sys

image=Image.open('coil-20-unproc/obj1__0.png')
image_array=np.array(image)
original_shape=image_array.shape
image_array=np.pad(image_array, pad_width=((0, (8-image_array.shape[0]%8)%8), (0, (8-image_array.shape[1]%8)%8)), mode='edge')

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

if len(sys.argv)>1:
    quantization_value=int(sys.argv[1])
else:
    quantization_value=25
quantization_matrix=np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])*quantization_value/25

quantized_image_array=two_dimensional_dct(image_array).T
# print(np.max(quantized_image_array), np.min(quantized_image_array))
result=np.zeros(quantized_image_array.shape[0]*quantized_image_array.shape[1])
for i in range(0, len(quantized_image_array), 8):
    for j in range(0, len(quantized_image_array.T), 8):
        # print(i, j)
        result[i*len(quantized_image_array.T)+8*j:i*len(quantized_image_array.T)+8*j+64]=(np.round(quantized_image_array[i:i+8, j:j+8]/quantization_matrix)).flatten()
# with open('quantized_image_array2.txt', 'w') as file:
#     for i in range(len(quantized_image_array)):
#         if i%8!=0:
#             continue
#         for j in range(len(quantized_image_array.T)):
#             if j%8==0:
#                 file.write(f"{quantized_image_array[i, j]} ")
#         file.write('\n')
l=len(result)
if l>64:
    for i in range(0, l-64, 64):
        result[l-64-i]-=result[l-128-i]
# Note that 0 is actually slightly better than 1 as default, since 1 -> 3 bits, 0 -> 1 bit later on
def rle(array):
    result=[array[0], 0]
    for value in array[1:]:
        if value==result[-2]:
            result[-1]+=1
        else:
            result.append(value)
            result.append(0)
    return np.array(result)
rle_result=rle(result)

class TrieNode:
    def __init__(self, value=1e10, frequency=0):
        self.frequency=frequency
        self.left=None
        self.right=None
        self.is_end_of_word=value

    def __lt__(self, other):
        return self.frequency<other.frequency   
    
class HuffmanTrie:
    def __init__(self, int_list, original_shape, width, height, filepath='compressed_file.txt'):
        self.root=None
        self.character_encoding={}
        result=self.encode(int_list)
        self.filepath=filepath
        with open(self.filepath, 'w') as file:
            file.write(f"{original_shape[0]} {original_shape[1]} {width} {height} {result}")
        # with open('encoding.json', 'w') as file:
        #     file.write(str(self.character_encoding))
        print(len(result))
        assert (self.decode(self.filepath)==int_list).all()
    
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
    
    def decode(self, filepath):
        with open(filepath, 'r') as file:
            original_width, original_height, width, height, string=file.read().split()
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
        return np.array(result)
ht=HuffmanTrie(rle(result[[i for i in range(len(result)) if i%64 != 0]]), original_shape, image_array.shape[0], image_array.shape[1])  

ht2=HuffmanTrie(rle(result[range(0, len(result), 64)]), original_shape, image_array.shape[0], image_array.shape[1], 'dc_compressed.txt')

def invert_RLE(array):
    result=[]
    for i in range(0, len(array), 2):
        result+=[array[i]]*(int(array[i+1])+1)
    return np.array(result)

def intercalate_64(arr1, arr2):
    assert len(arr1)==len(arr2)*63
    result=np.zeros(len(arr1)+len(arr2))
    for i in range(0, len(result), 64):
        result[i]=arr2[i//64]
        result[i+1:i+64]=arr1[(i//64)*63:(i//64)*63+63]
    return np.array(result)

inverted_result=intercalate_64(invert_RLE(ht.decode(ht.filepath)), invert_RLE(ht2.decode(ht2.filepath)))

if l>64:
    for i in range(0, l-64, 64):
        inverted_result[i+64]+=inverted_result[i]
reshaped_inverted_result=np.zeros_like(quantized_image_array)
for i in range(0, len(reshaped_inverted_result), 8):
    for j in range(0, len(reshaped_inverted_result.T), 8):
        reshaped_inverted_result[i:i+8, j:j+8]=inverted_result[i*len(reshaped_inverted_result.T)+8*j:i*len(reshaped_inverted_result.T)+8*j+64].reshape(8, 8)
inverted_result=reshaped_inverted_result
# TODO -> Implement Inverse DCT, figure out the quantization matrix and plot the error vs byte storage. Try to implement Lempel-Ziv for accurate compression.
def inverse_DCT(array, ortho=True):
    if not ortho:
        norm='backward'
    else:
        norm='ortho'
    for i in range(0, len(array), 8):
        for j in range(0, len(array.T), 8):
            array[i:i+8, j:j+8]=idct(idct(array[i:i+8, j:j+8].T, norm=norm).T, norm=norm)
    return array

for i in range(0, len(quantized_image_array), 8):
    for j in range(0, len(quantized_image_array.T), 8):
        inverted_result[i:i+8, j:j+8]*=quantization_matrix

final_result=inverse_DCT(inverted_result)[0:original_shape[0], 0:original_shape[1]]
image_array=image_array[0:original_shape[0], 0:original_shape[1]]
print(np.sqrt(np.mean((image_array-final_result)**2)))

if len(sys.argv)==1:
    reconstructed_image=Image.fromarray(final_result)
    reconstructed_image.show()
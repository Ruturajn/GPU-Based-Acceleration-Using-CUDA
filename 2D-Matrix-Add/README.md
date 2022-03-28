# 2D Matrix Addition

This code performs 2D matrix addition, by using 3 different kernels, which are ```per_row_kernel```, ```per_column_kernel``` and ```per_element_kernel```,
where one thread handles a single row in the matrix, a single column in the matrix and a single element in the matrix respectively.

# Execution of the Code (In a Linux based OS):

## For Add_Matrix.cu : 

```
$ cd </path/to/the/location/of/the/program>
$ make
$ ./one.out
```

## For Add_Matrix_v2.cu : 

```
$ cd </path/to/the/location/of/the/program>
$ make FILE=two
$ ./two.out
```

**Note: You might want to change the architecture specific flags in the Makefile if your GPU has a different compute and sm architecture.**

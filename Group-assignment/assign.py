#!/usr/bin/env python3
# assign.py : part 3. Group assignments
# Modified by: Yuhan Zeng (yuhzeng)
# Teammates: Hu Hai, Surya Prateek Soni, Yuhan Zeng
# Last modified date: 9/27/2018
#
'''
Description: Local Search Algorithm is used in this solution.
The input file is read and transformed into a 2-D list: each row is the input of a student
in the format of ['djcran', '3', 'zehzhang,chen464', 'kapadia'] or ['chen464', '1', '_', '_'].
random_assign() function generates a random number <size> in set {1, 2, 3} every time as a group size, and then 
randomly choose <size> rows from the input list, assign them into one group, and analyze the time needed for
this group. Repeating generating groups of size 1 to size 3 and randomly choose students from the list until all
students are assigned a group.
Repetitively call random_assign() for <repeats> times and find the assignment with the lowest total time.

Result is more accurate with a higher <repeats> number. 
'''

import sys
import random
import math

def random_assign(students_input, k, m, n):
    num_students = len(students_input)
    row_set = set(range(0, num_students)) # A set of all the row numbers in the student_input array
    used_rows = set() # A set for storing row numbers that are already used
    num_groups = 0
    total_time = 0
    groups = [] # A list to store the final assigned groups
    
    while (len(used_rows) != len(students_input)):
        size = random.randint(1,3) # Generate a random integer in [1,3] for the current group size
        while(size > len(row_set)): # Make sure size is not larger than the row_set
            size = random.randint(1,3)
        group_rows = random.sample(row_set, size) # Randomly pick (size) students to put in one gorup (random.sample() returns a list)
        students_in_group = [] # A list to store the ID of all students in one group
        
        for i in range(size):
            students_in_group += [students_input[group_rows[i]][0]]
        groups.append(students_in_group)
       
        for i in range(size): # Start checking each student in this group
            row = group_rows[i]
            if (int(students_input[row][1]) != 0 \
                and int(students_input[row][1]) != size): # Check if the student's group size is same as his/her choice
                total_time += 1
            wanted = students_input[row][2].split(',') # Get a list of the student's wanted people
            if (wanted[0] != '_'): # If input is not empty
                if (len(wanted) == 1): # If there is only one wanted person
                    if not (wanted[0] in students_in_group):
                        total_time += n
                else: # If there are two wanted persons
                    if not (wanted[0] in students_in_group \
                            and wanted[1] in students_in_group): # If this student didn't get either of the two wanted students
                        total_time += n
            unwanted = students_input[row][3].split(',') #  Get a list of the student's unwanted people
            if (unwanted[0] != '_'):
                if (unwanted[0] in students_in_group): # If the student got at least one unwanted person
                    total_time += m
                if (len(unwanted) == 2 and unwanted[1] in students_in_group): # If the student got a second person that he didn't want
                    total_time += m
       
        used_rows.update(group_rows) # Put the assigned rows into used_rows set
        num_groups += 1 # Increment num of groups
        row_set = row_set - used_rows # Exclude the used rows from the set of rows
   
    total_time += num_groups * k
    return (groups, total_time)

# test cases
# Read input file
students_input = []
with open(sys.argv[1], 'r') as file:
    for line in file:
        students_input.append(line.split())
k = int(sys.argv[2]) # k minutes to grade each assignment
m = int(sys.argv[3]) # m minutes to deal with each student who didn't get the people the wanted
n = int(sys.argv[4]) # n minutes to deal with each student who got a person they didn't want

# The number of times repetitively calling random_assign()
# If the input file is large and the code runs slowly, you may properly decrease <repeats>
repeats = 50000

total = math.inf
assignments = []
# Call random_assign() for <repeats> times, and choose the assignment with the lowest total time
for i in range(repeats + 1):
    (groups, time) = random_assign(students_input, k, m, n)
    if time < total:
        total = time
        assignments = groups
for i in range(len(assignments)):
    s = ""
    for j in range(len(assignments[i])-1):
        s += assignments[i][j] + " "
    s += assignments[i][-1]
    print(s)
print(total)
        
    
                    
            
            
            
        
        
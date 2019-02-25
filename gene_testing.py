"""
Some testing for the genetic algorithm.

Author:
Sami Karkinen
"""
from random import random
import numpy as np

class Genetic_invidual:
    """
    Class for the evolving invidual for genetic algorithm
    """
    def __init__(self, params, val_mut, perc_mut):
        """
        The constructor

        Parameters
        ----------
        params - list
          The initial parameters for the invidual. The most important thing is
          that the list is the right length.
        val_mut - float
          The maximum percentage the value can change, likely good to have 0-1.
          1 means the value can double or be reduced to 0 while 0 prevents changes
          for the values.
        perc_mut - float
          The maximum percentage the amount of variables can change at a time.
          Use with values of 0-1.
        """
        self.params = params
        self.val_mut = val_mut
        self.perc_mut = perc_mut
        self.fitness = 0

    def mutate(self):
        """
        Mutate function. Mutates the invidual by the initialised values.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Just changes the values within the class.
        """
        len_mut = int(random()*self.perc_mut*len(self.params))
        new_params = self.params[:]
        for i in range(len_mut):
            i_mut = int(random()*len(self.params))
            if self.params[i_mut] != -1:
                if self.params[i_mut] == 0:
                    self.params[i_mut] = 0.5
                new_params[i_mut] = self.params[i_mut]*(1 - self.val_mut +\
                    2*random()*self.val_mut)
                self.params[i_mut] = -1
            else:
                i -= 1
        self.params = new_params

    # Definitions for sorting, making the class sortable
    def _is_valid_operand(self, other):
        return (hasattr(other, "fitness"))
    def __eq__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return (self.fitness == other.fitness)
    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return (self.fitness > other.fitness)

def mate(invidual1, invidual2):
    """
    2 inviduals mate and make 1 separate invidual

    Parameters
    ----------
    invidual1 : Genetic_invidual
      The first invidual for mating

    invidual2 : Genetic_invidual
      The second invidual for mating
    """

    new_invidual_params = invidual2.params[:]
    split_rate = random()*0.4+0.3
    for i in range(int(len(invidual1.params)*split_rate)):
        parameter_index = int(random()*len(invidual1.params))
        new_invidual_params[parameter_index] = invidual1.params[parameter_index]
    return Genetic_invidual(new_invidual_params, invidual1.val_mut, invidual1.perc_mut)

def survival_of_the_fittest(list_of_inviduals, kill_percent):
    """
    Kills the least fit inviduals and adds a combination of the best ones to the
    list.

    Parameters
    ----------
    list_of_inviduals : list
      The list of inviduals

    kill_percent : float
      The percentage of the inviduals in the list to be killed

    Returns
    -------
    list_of_inviduals
      The list of inviduals
    """
    list_of_inviduals.sort()
    kill_split = int(len(list_of_inviduals)*kill_percent)
    for i in range(kill_split):
        list_of_inviduals.pop(-i-1)

    for i in range(kill_split-2):
        list_of_inviduals.append(copy_invidual(list_of_inviduals[i]))

    list_of_inviduals = [mate(list_of_inviduals[0],list_of_inviduals[1])] + list_of_inviduals
    list_of_inviduals = [mate(list_of_inviduals[0],list_of_inviduals[2])] + list_of_inviduals

    for i in range(3, len(list_of_inviduals)):
        list_of_inviduals[i].mutate()

    return list_of_inviduals


def copy_invidual(invidual):
    """
    Copies the invidual and returns the copy
    """
    new_invidual = Genetic_invidual(invidual.params[:], invidual.val_mut, invidual.perc_mut)
    return new_invidual

if __name__ == "__main__":
    print("Genetic invidual, genetical algorithm testing:")
    amount_of_inviduals = 10
    number_of_cycles = 100
    inv_list = [Genetic_invidual([1,1,1], 0.3, 0.7)\
     for i in range(amount_of_inviduals)]

    print("A simple fitness function of params [1,1,1] getting to 2 from the initial 1.\n"\
     "Starting test, mutating inviduals...")

    # The fitness function: distance from 2
    for j in range(number_of_cycles):
        for i in inv_list:
            i.fitness = 0
            for k in i.params:
                if k > 2:
                    i.fitness -= 2-(k-2)
                else:
                    i.fitness += k

        inv_list = survival_of_the_fittest(inv_list, 0.4)
        print("Cycle:", j, "  The top fitness:", max([i.fitness for i in inv_list]))

    print("Test over")

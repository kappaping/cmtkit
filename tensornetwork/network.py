"""
Network class: Define the class network for the manipulation of tensor networks.
"""


import numpy as np




class network():
    # A class of tensor networks.
    # ----------
    # Attributes:
    # diagram: A map of the form {tensor name (str): bonds (list of int)}.
    # order: A list of string, where the elements are the tensor names in reversed order of contraction.
    # ext_bonds: A list of int, where the elements are the exterior bond indices of the whole tensor network.
    # tensors: A map of the form {tensor name (str): tensor (numpy.ndarray)}.
    # ----------

    
    def __init__(self, diagram_name="", tensors={}):
        self.set_diagram(diagram_name)
        self.tensors = tensors.copy()


    def set_diagram(self, diagram_name):
        """
        Set diagram function: Set up the tensor network diagram given a .dgm file.
        """

        diagram = {}
        ext_bonds = []
        order = []
        if diagram_name != "":
            with open(diagram_name, "r") as file_read:
                lines = file_read.readlines()
                for line in lines:
                    key, element = line.split(":")
                    if key == "order":
                        order = element.split()
                    elif key == "ext":
                        ext_bonds = list(map(int, element.strip().split()))
                    else:
                        diagram[key] = list(map(int, element.strip().split()))
        self.diagram = diagram
        self.ext_bonds = ext_bonds
        self.order = order


    def set_tensors(self, tensors_given):
        """
        Set tensors function: Set up the tensors in the tensor network from the given tensors.
        """

        for key in tensors_given.keys():
            self.tensors[key] = tensors_given[key]


    def contract(self, to_print=False):
        """
        Contract function: Contract the tensor network.
        """

        diagram = self.diagram
        order = self.order
        tensors = self.tensors
        # Check if there are tensors which should be self contracted.
        for name in order:
            bonds = diagram[name]
            while max([bonds.count(bond) for bond in bonds]) > 1:
                for i in range(len(bonds) - 1):
                    if bonds[i] in bonds[i + 1:]:
                        bond = bonds[i]
                        id0, id1 = i, i + 1 + bonds[i + 1:].index(bond)
                        tensors[name] = np.trace(tensors[name], axis1=id0, axis2=id1)
                        for j in range(2):
                            bonds.remove(bond)
                        diagram[name] = bonds
                        break
        # Contract the tensors sequentially.
        if to_print:
            print("Initial order:", order)
            print("    Diagram:", diagram)
            print("    Shapes:", {key: tensors[key].shape for key in tensors.keys()})
        while len(order) > 1:
            names = order[-2:]
            tensor_0, tensor_1 = [tensors[name] for name in names]
            bonds_0, bonds_1 = [diagram[name] for name in names]
            bonds_contract = list(set(bonds_0) & set(bonds_1))
            bonds_contract_ids = np.array([[bonds_0.index(bond), bonds_1.index(bond)] for bond in bonds_contract]).T
            tensor_new = np.tensordot(tensor_0, tensor_1, bonds_contract_ids)
            for bond in bonds_contract:
                bonds_0.remove(bond)
                bonds_1.remove(bond)
            bonds_new = bonds_0 + bonds_1
            for name in names:
                del diagram[name]
                del tensors[name]
                order.remove(name)
            name_new = names[0] + names[1]
            order.append(name_new)
            diagram[name_new] = bonds_new
            tensors[name_new] = tensor_new
            if to_print:
                print("Order:", order)
                print("    Diagram:", diagram)
                print("    Shapes:", {key: tensors[key].shape for key in tensors.keys()})
        ext_bonds = self.ext_bonds
        bond_ids = [diagram[order[0]].index(bond) for bond in ext_bonds]
        tensor = np.moveaxis(tensors[order[0]], bond_ids, range(len(bond_ids)))

        return tensor



        

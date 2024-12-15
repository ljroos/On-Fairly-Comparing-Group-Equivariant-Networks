"""
Dihedral group of order 8
https://proofwiki.org/wiki/Definition:Cyclic_Group

e = identity
a = 90 degree rotation  (x.rot90(k=1, dims=(-2, -1)))
b = flip width (left becomes right)  (x.flip(dims=(-1,)))

group actions are applied to the RIGHT of the object: applying g to x is written as
x * g. So for example ba^2 will first flip and then rotate 180 degrees.
"""

group_members = ["e", "a", "a^2", "a^3", "b", "ba", "ba^2", "ba^3"]

group_member_idx = {group_members[i]: i for i in range(len(group_members))}
idx_to_group_member = {i: group_members[i] for i in range(len(group_members))}

# instead of chaining .flip() and .rot90(), which make an unnecessary copy of the tensor, .transpose() is used, since it does not make an unecessary copy.
# Note: z2 refers to pairs of integers. Z2 refers to the group of integers modulo 2.
# f_z2 refers to functions on z2 (i.e. images).
D4_action_on_f_z2 = {
    "e": lambda x: x,
    "a": lambda x: x.rot90(k=1, dims=(-2, -1)),
    "a^2": lambda x: x.rot90(k=2, dims=(-2, -1)),
    "a^3": lambda x: x.rot90(k=3, dims=(-2, -1)),
    "b": lambda x: x.flip(dims=(-1,)),
    "ba": lambda x: x.transpose(-2, -1),
    "ba^2": lambda x: x.flip(dims=(-2,)),
    "ba^3": lambda x: x.transpose(-2, -1).rot90(
        k=2, dims=(-2, -1)
    ),  # anti-diagonal transpose.
}  # consider adding .contiguous() to "ba" action, if problems arise.

D4_action_on_trivial = {member: lambda x: x for member in group_members}

D4_actions = {"z2": D4_action_on_f_z2, "trivial": D4_action_on_trivial}


"""
Use the Cayley table of D4, in order to calculate the group dimension permutation of each group member.

Explanation: the Cayley table says G_i*G_j = Table[i, j].

What we want is an inverse table T s.t. T[i, j] * G_i = G_j. Use the Cayley table to calculate T[i, j] = G_j * G_i^-1.

For D4 an inverse table is calculated using code, the rest are just written down as illustrative examples.

https://proofwiki.org/wiki/Dihedral_Group_D4/Subgroups
"""

# Dihedral group of order 8 Cayley table
D4_cayley_table = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 0, 7, 4, 5, 6],
    [2, 3, 0, 1, 6, 7, 4, 5],
    [3, 0, 1, 2, 5, 6, 7, 4],
    [4, 5, 6, 7, 0, 1, 2, 3],
    [5, 6, 7, 4, 3, 0, 1, 2],
    [6, 7, 4, 5, 2, 3, 0, 1],
    [7, 4, 5, 6, 1, 2, 3, 0],
]

# Use code to generate the inverse table.
inverses = [0, 3, 2, 1, 4, 5, 6, 7]
D4_table = [[D4_cayley_table[j][inverses[i]] for j in range(8)] for i in range(8)]


# Cyclic group of order 4
# https://proofwiki.org/wiki/Definition:Cyclic_Group
C4_cayley_table = [
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
]

# note how for the inverse table, T[i, j] * G_i = G_j.
C4_table = [
    [0, 1, 2, 3],
    [3, 0, 1, 2],
    [2, 3, 0, 1],
    [1, 2, 3, 0],
]

# Klein 4 group
# https://proofwiki.org/wiki/Definition:Klein_Four-Group
K4_cayley_table = [
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0],
]

# note for K4, the Cayley table is the same as the inverse table!
K4_table = [
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0],
]

# Cyclical group of order 2
# all 2 element groups are isomorphic to this one.
C2_table = [
    [0, 1],
    [1, 0],
]

# Trivial group
C1_table = [
    [0],
]

"""
Define all 10 subgroups of D4 by its group members and Cayley table.

https://proofwiki.org/wiki/Dihedral_Group_D4/Subgroups

Names are selected intuitively, based on torch operations used to define actions of the group members.
"""

D4_subgroups = {
    "D4": {
        "name": "D4",
        "members": ["e", "a", "a^2", "a^3", "b", "ba", "ba^2", "ba^3"],
        "table": D4_table,
    },
    "trivial": {
        "name": "trivial",
        "members": ["e"],
        "table": C1_table,
    },
    "rot90": {
        "name": "rot90",
        "members": ["e", "a", "a^2", "a^3"],
        "table": C4_table,
    },
    "rot180": {
        "name": "rot180",
        "members": ["e", "a^2"],
        "table": C2_table,
    },
    "flipW": {
        "name": "flipW",
        "members": ["e", "b"],
        "table": C2_table,
    },
    "transpose": {
        "name": "transpose",
        "members": ["e", "ba"],
        "table": C2_table,
    },
    "flipH": {
        "name": "flipH",
        "members": ["e", "ba^2"],
        "table": C2_table,
    },
    "antidiagonal_transpose": {
        "name": "antidiagonal_transpose",
        "members": ["e", "ba^3"],
        "table": C2_table,
    },
    "flipH_and_or_flipW": {
        "name": "flipH_and_or_flipW",
        "members": ["e", "a^2", "b", "ba^2"],
        "table": K4_table,
    },
    "rot180_and_or_transpose": {
        "name": "rot180_and_or_transpose",
        "members": ["e", "a^2", "ba", "ba^3"],
        "table": K4_table,
    },
}


# will change when different in/out groups are supported.
def old_get_transformation_law(group_name: str, action_space: str):
    supported_groups = D4_subgroups.keys()
    assert (
        group_name in supported_groups
    ), f"group {group_name} not supported. Supported groups: {supported_groups}"
    group = D4_subgroups[group_name]
    members = group["members"]
    table = group["table"]
    action = [D4_actions[action_space][members[g]] for g in range(len(members))]
    return lambda x, g: action[g](x)[:, :, table[g]]


def get_transformation_law(in_group: str, out_group: str, action_space: str):
    # TODO: more general implementation for when in/out group are different.
    supported_groups = D4_subgroups.keys()
    assert (
        in_group in supported_groups
    ), f"in_group {in_group} not supported. Supported groups: {supported_groups}"
    assert (
        out_group in supported_groups
    ), f"out_group {out_group} not supported. Supported groups: {supported_groups}"

    if in_group != out_group and in_group != "trivial":
        raise NotImplementedError(
            f"Only trivial group can be used as in_group for now. Got {in_group}"
        )

    # TODO: temporary hack for the trivial group
    if in_group == out_group:
        law = old_get_transformation_law(in_group, action_space)
        return law, law
    else:
        group = out_group
        group = D4_subgroups[group]
        members = group["members"]
        action = [D4_actions[action_space][members[g]] for g in range(len(members))]
        in_law = lambda x, g: action[g](x)
        out_law = old_get_transformation_law(out_group, action_space)
        return in_law, out_law


if __name__ == "__main__":
    import torch

    # verify that actions are defined correctly.
    true_members_f_z2_action = {
        "e": lambda x: x,
        "a": lambda x: x.rot90(k=1, dims=(-2, -1)),
        "a^2": lambda x: x.rot90(k=2, dims=(-2, -1)),
        "a^3": lambda x: x.rot90(k=3, dims=(-2, -1)),
        "b": lambda x: x.flip(dims=(-1,)),
        "ba": lambda x: x.flip(dims=(-1,)).rot90(k=1, dims=(-2, -1)),
        "ba^2": lambda x: x.flip(dims=(-1,)).rot90(k=2, dims=(-2, -1)),
        "ba^3": lambda x: x.flip(dims=(-1,)).rot90(k=3, dims=(-2, -1)),
    }

    # Test D4 actions.
    x = torch.arange(1 * 1 * 3 * 3).reshape(1, 1, 3, 3)

    all_correct = True
    for n, g in enumerate(true_members_f_z2_action.keys()):
        action1, action2 = (
            true_members_f_z2_action[g],
            D4_action_on_f_z2[g],
        )
        y1, y2 = action1(x), action2(x)
        equal = y1.allclose(y2)
        if not equal:
            print(f"{n}: {equal}")
            print(x)
            print(y1)
            print(y2)
            all_correct = False

    if all_correct:
        print("D4 group actions are correct!")
    else:
        print("Some errors!")

    # Test D4 group Cayley table.
    all_correct = True
    for g1 in group_member_idx.keys():
        for g2 in group_member_idx.keys():
            # check if applying two actions in a row is the same as applying the action of the product of the two actions
            i, j = group_member_idx[g1], group_member_idx[g2]
            g3 = idx_to_group_member[D4_cayley_table[i][j]]

            y1 = D4_action_on_f_z2[g2](D4_action_on_f_z2[g1](x))
            y2 = D4_action_on_f_z2[g3](x)

            equal = y1.allclose(y2)
            if not equal:
                print(f"{g1, g2}: {equal}")
                print(x)
                print(y1)
                print(y2)
                all_correct = False

    if all_correct:
        print("D4 group Cayley table is correct!")
    else:
        print("Some errors!")

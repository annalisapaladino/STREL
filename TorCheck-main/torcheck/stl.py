#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# Copyright 2020-* Luca Bortolussi. All Rights Reserved.
# Copyright 2020-* Laura Nenzi.     All Rights Reserved.
# Copyright 2020-* AI-CPS Group @ University of Trieste. All Rights Reserved.
# ==============================================================================

"""A fully-differentiable implementation of Signal Temporal Logic semantic trees."""

from typing import Union

# For custom type-hints
# For tensor functions
import torch
import torch.nn.functional as F
from torch import Tensor

# Custom types
realnum = Union[float, int] # Defines a custom type alias for any real number (used for thresholds)


# TODO: automatic check of timespan when evaluating robustness? (should be done only at root node)

def eventually(x: Tensor, time_span: int) -> Tensor:
    # TODO: as of this implementation, the time_span must be int (we are working with steps,
    #  not exactly points in the time axis)
    # TODO: maybe converter from resolution to steps, if one has different setting
    """
    STL operator 'eventually' in 1D.

    Parameters
    ----------
    x: torch.Tensor
        Signal
    time_span: any numeric type
        Timespan duration

    Returns
    -------
    torch.Tensor
    A tensor containing the result of the operation.
    """
    return F.max_pool1d(x, kernel_size=time_span, stride=1)
    # Defines an eventually operator using 1D max pooling:
    #    Applies over time.
    #    For each window of time_span, takes the maximum value.
    #    Used in quantitative semantics to implement F and G

class Node:
    """Abstract node class for STL semantics tree."""

    def __init__(self) -> None:
        # Must be overloaded.
        pass

    def __str__(self) -> str:
        # Must be overloaded.
        pass

    def boolean(self, x: Tensor, evaluate_at_all_times: bool = False) -> Tensor:
        """
        Evaluates the boolean semantics at the node.

        Parameters
        ----------
        x : torch.Tensor, of size N_samples x N_vars x N_sampling_points
            The input signals, stored as a batch tensor with trhee dimensions.
        evaluate_at_all_times: bool
            Whether to evaluate the semantics at all times (True) or
            just at t=0 (False).

        Returns
        -------
        torch.Tensor
        A tensor with the boolean semantics for the node.
        """
        z: Tensor = self._boolean(x)
        if evaluate_at_all_times:
            return z
        else:
            return self._extract_semantics_at_time_zero(z)

        # Calls internal _boolean method to evaluate Boolean semantics.
        # By default returns value at time t=0 unless evaluate_at_all_times=True.

    def quantitative(
            self,
            x: Tensor,
            normalize: bool = False,
            evaluate_at_all_times: bool = False,
    ) -> Tensor:
        """
        Evaluates the quantitative semantics at the node.

        Parameters
        ----------
        x : torch.Tensor, of size N_samples x N_vars x N_sampling_points
            The input signals, stored as a batch tensor with three dimensions.
        normalize: bool
            Whether the measure of robustness if normalized (True) or
            not (False). Currently not in use.
        evaluate_at_all_times: bool
            Whether to evaluate the semantics at all times (True) or
            just at t=0 (False).

        Returns
        -------
        torch.Tensor
        A tensor with the quantitative semantics for the node.
        """
        z: Tensor = self._quantitative(x, normalize) # normalize is a placeholder
        if evaluate_at_all_times:
            return z
        else:
            return self._extract_semantics_at_time_zero(z)
        
        # Same idea as boolean(...) but for robustness evaluation.
        
    def set_normalizing_flag(self, value: bool = True) -> None:
        """
        Setter for the 'normalization of robustness of the formula' flag.
        Currently not in use.
        """

    def time_depth(self) -> int:
        """Returns time depth of bounded temporal operators only."""
        # Must be overloaded.

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        """Private method equivalent to public one for inner call."""
        # Must be overloaded.

    def _boolean(self, x: Tensor) -> Tensor:
        """Private method equivalent to public one for inner call."""
        # Must be overloaded.

    @staticmethod
    def _extract_semantics_at_time_zero(x: Tensor) -> Tensor:
        """Extrapolates the vector of truth values at time zero"""
        return torch.reshape(x[:, 0, 0], (-1,))


class Atom(Node):
    """Atomic formula node; for now of the form X<=t or X>=t"""

    def __init__(self, var_index: int, threshold: realnum, lte: bool = False) -> None:
        super().__init__()
        self.var_index: int = var_index
        self.threshold: realnum = threshold
        self.lte: bool = lte

    def __str__(self) -> str:
        s: str = (
                "x_"
                + str(self.var_index)
                + (" <= " if self.lte else " >= ")
                + str(round(self.threshold, 4))
        )
        return s

    def time_depth(self) -> int:
        return 0

    def _boolean(self, x: Tensor) -> Tensor: # Boolean: checks if x[var_index] <= threshold
        # extract tensor of the same dimension as data, but with only one variable
        xj: Tensor = x[:, self.var_index, :]
        xj: Tensor = xj.view(xj.size()[0], 1, -1)
        if self.lte:
            z: Tensor = torch.le(xj, self.threshold)
        else:
            z: Tensor = torch.ge(xj, self.threshold)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor: # Quantitative: returns the signed distance from the threshold (e.g. how much above or below)
        # extract tensor of the same dimension as data, but with only one variable
        xj: Tensor = x[:, self.var_index, :]
        xj: Tensor = xj.view(xj.size()[0], 1, -1)
        if self.lte:
            z: Tensor = -xj + self.threshold
        else:
            z: Tensor = xj - self.threshold
        if normalize:
            z: Tensor = torch.tanh(z)
        return z


class Not(Node):
    """Negation node."""

    def __init__(self, child: Node) -> None:
        super().__init__()
        self.child: Node = child

    def __str__(self) -> str:
        s: str = "not ( " + self.child.__str__() + " )"
        return s

    def time_depth(self) -> int:
        return self.child.time_depth()

    def _boolean(self, x: Tensor) -> Tensor:
        z: Tensor = ~self.child._boolean(x)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z: Tensor = -self.child._quantitative(x, normalize) # negates the robustness score
        return z


class And(Node):
    """Conjunction node."""

    def __init__(self, left_child: Node, right_child: Node) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child

    def __str__(self) -> str:
        s: str = (
                "( "
                + self.left_child.__str__()
                + " and "
                + self.right_child.__str__()
                + " )"
        )
        return s

    def time_depth(self) -> int:
        return max(self.left_child.time_depth(), self.right_child.time_depth())

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.left_child._boolean(x)
        z2: Tensor = self.right_child._boolean(x)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.logical_and(z1, z2)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.left_child._quantitative(x, normalize)
        z2: Tensor = self.right_child._quantitative(x, normalize)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.min(z1, z2)
        return z


class Or(Node):
    """Disjunction node."""

    def __init__(self, left_child: Node, right_child: Node) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child

    def __str__(self) -> str:
        s: str = (
                "( "
                + self.left_child.__str__()
                + " or "
                + self.right_child.__str__()
                + " )"
        )
        return s

    def time_depth(self) -> int:
        return max(self.left_child.time_depth(), self.right_child.time_depth())

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.left_child._boolean(x)
        z2: Tensor = self.right_child._boolean(x)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.logical_or(z1, z2)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.left_child._quantitative(x, normalize)
        z2: Tensor = self.right_child._quantitative(x, normalize)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.max(z1, z2)
        return z


class Globally(Node): # ALWAYS operator: the formula is always true in the time span
    """Globally node."""

    def __init__(
            self,
            child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
            adapt_unbound: bool = True,
    ) -> None:
        super().__init__()
        self.child: Node = child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1
        self.adapt_unbound: bool = adapt_unbound

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "always" + s0 + " ( " + self.child.__str__() + " )"
        return s

    def time_depth(self) -> int:
        if self.unbound:
            return self.child.time_depth()
        elif self.right_unbound:
            return self.child.time_depth() + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return self.child.time_depth() + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.child._boolean(x[:, :, self.left_time_bound:])  # nested temporal parameters
        # z1 = z1[:, :, self.left_time_bound:]
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummin(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.min(z1, 2, keepdim=True)
        else:
            z: Tensor = torch.ge(1.0 - eventually((~z1).double(), self.right_time_bound - self.left_time_bound), 0.5)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.child._quantitative(x[:, :, self.left_time_bound:], normalize)
        # z1 = z1[:, :, self.left_time_bound:]
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummin(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.min(z1, 2, keepdim=True)
        else:
            z: Tensor = -eventually(-z1, self.right_time_bound - self.left_time_bound)
        return z


class Eventually(Node): # EVENTUALLY operator: the formula is eventually true (true at any time) in the time span
    """Eventually node."""

    def __init__(
            self,
            child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
            adapt_unbound: bool = True,
    ) -> None:
        super().__init__()
        self.child: Node = child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1
        self.adapt_unbound: bool = adapt_unbound

        if (self.unbound is False) and (self.right_unbound is False) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "eventually" + s0 + " ( " + self.child.__str__() + " )"
        return s

    # TODO: coherence between computation of time depth and time span given when computing eventually 1d
    def time_depth(self) -> int:
        if self.unbound:
            return self.child.time_depth()
        elif self.right_unbound:
            return self.child.time_depth() + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return self.child.time_depth() + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.child._boolean(x[:, :, self.left_time_bound:])
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummax(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.max(z1, 2, keepdim=True)
        else:
            z: Tensor = torch.ge(eventually(z1.double(), self.right_time_bound - self.left_time_bound), 0.5)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.child._quantitative(x[:, :, self.left_time_bound:], normalize)
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummax(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.max(z1, 2, keepdim=True)
        else:
            z: Tensor = eventually(z1, self.right_time_bound - self.left_time_bound)
        return z


class Until(Node):
    # UNTIL operator: phi_1 U[a, b] phi_2: 
    #   phi_2 must be true sometime in [a, b]
    #   phi_1 must be true at all time before that
    
    # TODO: maybe define timed and untimed until, and use this class to wrap them
    # TODO: maybe faster implementation (of untimed until especially)
    """Until node."""

    def __init__(
            self,
            left_child: Node, # phi_1
            right_child: Node, # phi_2
            unbound: bool = False, # phi_1 U phi_2 over infinite time
            right_unbound: bool = False, # only upper bound is infinite
            left_time_bound: int = 0, # define the interval [a, b]
            right_time_bound: int = 1, # define the interval [a, b]
    ) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child
        self.unbound: bool = unbound # Unbounded: full future
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1

        if (self.unbound is False) and (self.right_unbound is False) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "( " + self.left_child.__str__() + " until" + s0 + " " + self.right_child.__str__() + " )"
        return s

    def time_depth(self) -> int: # Calculates how far in time this operator needs to look to compute its output
        sum_children_depth: int = self.left_child.time_depth() + self.right_child.time_depth()
        if self.unbound:
            return sum_children_depth
        elif self.right_unbound:
            return sum_children_depth + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return sum_children_depth + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _boolean(self, x: Tensor) -> Tensor:
        if self.unbound:
            # this is phi_1 U phi_2 over the full future: 
            # We compute, for each time step, whether phi_2 becomes true in the future, and phi_1 holds up until then 
            z1: Tensor = self.left_child._boolean(x) # Get Boolean values for both operands
            z2: Tensor = self.right_child._boolean(x)
            # Shape: [batch, 1, time]
            size: int = min(z1.size()[2], z2.size()[2]) 
            z1: Tensor = z1[:, :, :size]
            z2: Tensor = z2[:, :, :size]
            
            # Builds a "history matrix" for each time step: was phi_1 true from now until t?
            z1_rep = torch.repeat_interleave(z1.unsqueeze(2), z1.unsqueeze(2).shape[-1], 2)
            z1_tril = torch.tril(z1_rep.transpose(2, 3), diagonal=-1)
            z1_triu = torch.triu(z1_rep)
            z1_def = torch.cummin(z1_tril + z1_triu, dim=3)[0]
            # same for z2
            z2_rep = torch.repeat_interleave(z2.unsqueeze(2), z2.unsqueeze(2).shape[-1], 2)
            z2_tril = torch.tril(z2_rep.transpose(2, 3), diagonal=-1)
            z2_triu = torch.triu(z2_rep)
            z2_def = z2_tril + z2_triu
            
            # Compute min(phi_1 segment, phi_2) across time window. Then take max over possible satisfaction points
            z: Tensor = torch.max(torch.min(torch.cat([z1_def.unsqueeze(-1), z2_def.unsqueeze(-1)], dim=-1), dim=-1)[0],
                                  dim=-1)[0] # This gives the truth value of the Until for each time point.
        elif self.right_unbound:
            # φ1 U[a,b] φ2 ≡ G[0,a](φ1) ∧ F[a,b](φ2) ∧ F[a,b](φ1 U φ2)
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, right_unbound=True,
                                                   left_time_bound=self.left_time_bound),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._boolean(x)
        else:
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, left_time_bound=self.left_time_bound,
                                                   right_time_bound=self.right_time_bound - 1),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._boolean(x)
            # It rewrites bounded until in terms of:
            #   Globally: ensures φ1 holds at start
            #   Eventually(φ2): ensures φ2 happens
            #   Eventually(UnboundedUntil): handles general satisfaction
        
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        if self.unbound:
            z1: Tensor = self.left_child._quantitative(x, normalize)
            z2: Tensor = self.right_child._quantitative(x, normalize)
            size: int = min(z1.size()[2], z2.size()[2])
            z1: Tensor = z1[:, :, :size]
            z2: Tensor = z2[:, :, :size]

            z1_rep = torch.repeat_interleave(z1.unsqueeze(2), z1.unsqueeze(2).shape[-1], 2)
            z1_tril = torch.tril(z1_rep.transpose(2, 3), diagonal=-1)
            z1_triu = torch.triu(z1_rep)
            z1_def = torch.cummin(z1_tril + z1_triu, dim=3)[0]

            z2_rep = torch.repeat_interleave(z2.unsqueeze(2), z2.unsqueeze(2).shape[-1], 2)
            z2_tril = torch.tril(z2_rep.transpose(2, 3), diagonal=-1)
            z2_triu = torch.triu(z2_rep)
            z2_def = z2_tril + z2_triu
            z: Tensor = torch.max(torch.min(torch.cat([z1_def.unsqueeze(-1), z2_def.unsqueeze(-1)], dim=-1), dim=-1)[0],
                                  dim=-1)[0]
            # z: Tensor = torch.cat([torch.max(torch.min(
            #    torch.cat([torch.cummin(z1[:, :, t:].unsqueeze(-1), dim=2)[0], z2[:, :, t:].unsqueeze(-1)], dim=-1),
            #    dim=-1)[0], dim=2, keepdim=True)[0] for t in range(size)], dim=2)
        elif self.right_unbound:
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, right_unbound=True,
                                                   left_time_bound=self.left_time_bound),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._quantitative(x, normalize=normalize)
        else:
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, left_time_bound=self.left_time_bound,
                                                   right_time_bound=self.right_time_bound - 1),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._quantitative(x, normalize=normalize)
        return z

class Since(Node):
    """Since node."""

    # SINCE operator: phi_1 U[a, b] phi_2: 
    #   phi_2 held within [a, b]
    #   phi_1 held from then until now

    # STL doesn’t natively support past, but we can simulate it by flipping time and reusing Until
    
    def __init__(
            self,
            left_child: Node,
            right_child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
    ) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1

        if (not self.unbound) and (not self.right_unbound) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = f"( {self.left_child} since{s0} {self.right_child} )"
        return s

    def time_depth(self) -> int:
        sum_children_depth: int = self.left_child.time_depth() + self.right_child.time_depth()
        if self.unbound:
            return sum_children_depth
        elif self.right_unbound:
            return sum_children_depth + self.left_time_bound
        else:
            return sum_children_depth + self.right_time_bound - 1

    def _boolean(self, x: Tensor) -> Tensor:
        # Past-time: need to flip the input
        x_flipped = torch.flip(x, [2]) # reverse along time axis

        # Reuse Until semantics on flipped signal
        until_node = Until( # construct a matching Until
            self.left_child,
            self.right_child,
            unbound=self.unbound,
            right_unbound=self.right_unbound,
            left_time_bound=self.left_time_bound,
            right_time_bound=self.right_time_bound - 1,
        )

        # Compute on flipped signal
        z_flipped = until_node._boolean(x_flipped)

        # Flip back
        return torch.flip(z_flipped, [2]) # flip back

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        # Past-time: need to flip the input
        x_flipped = torch.flip(x, [2])

        until_node = Until(
            self.left_child,
            self.right_child,
            unbound=self.unbound,
            right_unbound=self.right_unbound,
            left_time_bound=self.left_time_bound,
            right_time_bound=self.right_time_bound - 1,
        )

        z_flipped = until_node._quantitative(x_flipped, normalize=normalize)

        return torch.flip(z_flipped, [2])

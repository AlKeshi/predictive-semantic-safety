from __future__ import annotations


class ReactiveContacts:
    def record_contacts(self):
        self.hazard_tick = self.environment_tick = False
        for con in self.data.contact[: self.data.ncon]:
            a, b = (int(con.geom1), int(con.geom2))
            if b in self.robot_geoms:
                a, b = (b, a)
            if a not in self.robot_geoms or b not in self.hazard_geoms | self.environment_geoms:
                continue
            kind = "hazard" if b in self.hazard_geoms else "environment"
            self.hazard_tick |= kind == "hazard"
            self.environment_tick |= kind == "environment"
            key = (a, b)
            if key not in self.first_pairs:
                self.first_pairs.add(key)
                self.contacts.append(
                    {
                        "time": float(self.data.time),
                        "kind": kind,
                        "robot_geom": self.model.geom(a).name,
                        "other_geom": self.model.geom(b).name,
                        "robot_geom_id": a,
                        "other_geom_id": b,
                        "robot_xy": self.data.qpos[:2].tolist(),
                        "contact_position": con.pos.tolist(),
                        "signed_distance_m": float(con.dist),
                    }
                )
        self.contact_ticks += self.hazard_tick
        self.environment_ticks += self.environment_tick

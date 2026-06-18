/*
 * Copyright (C) 2026 Amlogic, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "symbol_table.h"

#include <fstream>
#include <sstream>

namespace {

std::string NormalizeBpeSymbol(std::string sym) {
    if (sym.size() >= 3) {
        const uint8_t *p = reinterpret_cast<const uint8_t *>(sym.c_str());
        if (p[0] == 0xe2 && p[1] == 0x96 && p[2] == 0x81) {
            return " " + sym.substr(3);
        }
    }
    return sym;
}

}  // namespace

SymbolTable::SymbolTable(const std::string &path) {
    std::ifstream is(path);
    if (!is.is_open()) {
        return;
    }

    std::string sym;
    int32_t id = 0;
    while (is >> sym >> id) {
        sym = NormalizeBpeSymbol(std::move(sym));
        id2sym_[id] = sym;
    }

    ok_ = !id2sym_.empty();
}

const std::string &SymbolTable::Lookup(int32_t id) const {
    static const std::string kEmpty;
    auto it = id2sym_.find(id);
    if (it == id2sym_.end()) {
        return kEmpty;
    }
    return it->second;
}

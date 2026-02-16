import { NavLink } from 'react-router-dom'

const navItems = [
  { to: '/', label: 'Detect', icon: '🔍' },
  { to: '/gallery', label: 'Gallery', icon: '🖼️' },
  { to: '/about', label: 'About', icon: 'ℹ️' },
]

export default function Layout({ children }) {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="bg-slate-800 border-b border-slate-700 px-4 py-3">
        <h1 className="text-xl font-bold text-clover-400 text-center">
          Clovr Fndr
        </h1>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">{children}</main>

      {/* Bottom Navigation */}
      <nav className="bg-slate-800 border-t border-slate-700 px-4 py-2">
        <div className="flex justify-around max-w-md mx-auto">
          {navItems.map(({ to, label, icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex flex-col items-center py-1 px-3 rounded-lg transition-colors ${
                  isActive
                    ? 'text-clover-400'
                    : 'text-slate-400 hover:text-slate-200'
                }`
              }
            >
              <span className="text-xl">{icon}</span>
              <span className="text-xs mt-0.5">{label}</span>
            </NavLink>
          ))}
        </div>
      </nav>
    </div>
  )
}
